"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence using the existing TTS pipeline.

Protocol:
    Client -> Server (TTS flow):
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input

    Client -> Server (voice management — usable at any time):
        {"type": "voice.list"}
        {"type": "voice.add", "name": "...", "consent": "...", ...}
        {"type": "voice.remove", "name": "..."}

    Server -> Client (TTS responses):
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        {"type": "audio.done", "sentence_index": 0}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}

    Server -> Client (voice management responses):
        {"type": "voice.list.result", "voices": [...], "uploaded_voices": [...]}
        {"type": "voice.add.result", "success": true, "voice": {...}}
        {"type": "voice.remove.result", "success": true, "name": "..."}

Voice management messages:
    voice.list: No additional fields required.

    voice.add: Accepts either audio or a pre-computed speaker embedding.
        With audio (base64):
            {"type": "voice.add", "name": "...", "consent": "...",
             "audio_data": "<base64>", "audio_format": "wav",
             "ref_text": "...", "speaker_description": "..."}
        With embedding:
            {"type": "voice.add", "name": "...", "consent": "...",
             "speaker_embedding": [0.1, 0.2, ...]}

    voice.remove:
        {"type": "voice.remove", "name": "..."}
"""

import asyncio
import base64
import io
import json
from contextlib import aclosing

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.datastructures import Headers, UploadFile
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.text_splitter import (
    SPLIT_CLAUSE,
    SPLIT_SENTENCE,
    SentenceSplitter,
)

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_PCM_SAMPLE_RATE = 24000
_MAX_CONFIG_MESSAGE_SIZE = 4 * 1024 * 1024  # allow large ref_audio payloads
_MAX_INPUT_TEXT_MESSAGE_SIZE = 128 * 1024
_MAX_VOICE_ADD_MESSAGE_SIZE = 16 * 1024 * 1024  # 16 MiB for base64 audio (~12 MB raw)
_VOICE_MSG_TYPES = frozenset({"voice.list", "voice.add", "voice.remove"})
_MIME_FROM_FORMAT = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "aac": "audio/aac",
    "webm": "audio/webm",
    "mp4": "audio/mp4",
}


class OmniStreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    Each WebSocket connection is an independent session. Text arrives
    incrementally, is split at sentence boundaries, and audio is generated
    per sentence using the existing OmniOpenAIServingSpeech pipeline.

    Args:
        speech_service: The existing TTS serving instance (reused for
            validation and audio generation).
        idle_timeout: Max seconds to wait for a message before closing.
        config_timeout: Max seconds to wait for the initial session.config.
    """

    def __init__(
        self,
        speech_service: OmniOpenAIServingSpeech,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    ) -> None:
        self._speech_service = speech_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            # 1. Wait for session.config
            config = await self._receive_config(websocket)
            if config is None:
                return  # Error already sent, connection closing

            # Validate model if specified
            if config.model and hasattr(self._speech_service, "_check_model"):
                error = await self._speech_service._check_model(
                    OpenAICreateSpeechRequest(input="ping", model=config.model)
                )
                if error is not None:
                    await self._send_error(websocket, str(error))
                    return

            boundary_re = SPLIT_CLAUSE if config.split_granularity == "clause" else SPLIT_SENTENCE
            splitter = SentenceSplitter(boundary_re=boundary_re)
            sentence_index = 0

            # 2. Receive text chunks until input.done (voice.* also accepted)
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    await self._send_error(websocket, "Idle timeout: no message received")
                    return

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                    continue

                if not isinstance(msg, dict):
                    await self._send_error(websocket, "WebSocket messages must be JSON objects")
                    continue

                msg_type = msg.get("type")

                if msg_type == "input.text":
                    if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
                        await self._send_error(websocket, "input.text message too large")
                        continue
                    text = msg.get("text", "")
                    if not isinstance(text, str):
                        await self._send_error(websocket, "input.text requires a string value")
                        continue
                    sentences = splitter.add_text(text)
                    for sentence in sentences:
                        await self._generate_and_send(websocket, config, sentence, sentence_index)
                        sentence_index += 1

                elif msg_type == "input.done":
                    remaining = splitter.flush()
                    if remaining:
                        await self._generate_and_send(websocket, config, remaining, sentence_index)
                        sentence_index += 1

                    await websocket.send_json(
                        {
                            "type": "session.done",
                            "total_sentences": sentence_index,
                        }
                    )
                    return

                elif msg_type in _VOICE_MSG_TYPES:
                    if msg_type == "voice.add" and len(raw) > _MAX_VOICE_ADD_MESSAGE_SIZE:
                        await self._send_error(websocket, "voice.add message too large")
                        continue
                    await self._handle_voice_message(websocket, msg)

                else:
                    await self._send_error(
                        websocket,
                        f"Unknown message type: {msg_type}",
                    )

        except WebSocketDisconnect:
            logger.info("Streaming speech: client disconnected")
        except Exception as e:
            logger.exception("Streaming speech session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                logger.debug("Failed to send error to streaming speech client", exc_info=True)

    async def _receive_config(self, websocket: WebSocket) -> StreamingSpeechSessionConfig | None:
        """Wait for and validate the session.config message.

        Voice management messages (voice.*) are handled inline while waiting
        for the config, so clients can manage voices before starting TTS.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self._config_timeout

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                await self._send_error(websocket, "Timeout waiting for session.config")
                return None

            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                await self._send_error(websocket, "Timeout waiting for session.config")
                return None

            max_size = _MAX_CONFIG_MESSAGE_SIZE
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid JSON")
                continue

            if not isinstance(msg, dict):
                await self._send_error(websocket, "Messages must be JSON objects")
                continue

            msg_type = msg.get("type")

            if msg_type in _VOICE_MSG_TYPES:
                if msg_type == "voice.add" and len(raw) > _MAX_VOICE_ADD_MESSAGE_SIZE:
                    await self._send_error(websocket, "voice.add message too large")
                    continue
                await self._handle_voice_message(websocket, msg)
                continue

            if len(raw) > max_size:
                await self._send_error(websocket, "session.config message too large")
                return None

            if msg_type != "session.config":
                await self._send_error(
                    websocket,
                    f"Expected session.config or voice.*, got: {msg_type}",
                )
                return None

            try:
                config = StreamingSpeechSessionConfig(**{k: v for k, v in msg.items() if k != "type"})
            except ValidationError as e:
                await self._send_error(websocket, f"Invalid session config: {e}")
                return None

            return config

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        sentence_index: int,
    ) -> None:
        """Generate audio for a single sentence and send it over WebSocket."""
        response_format = config.response_format or "wav"

        request = OpenAICreateSpeechRequest(
            input=sentence_text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format=response_format,
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=config.stream_audio,
        )

        start_payload = {
            "type": "audio.start",
            "sentence_index": sentence_index,
            "sentence_text": sentence_text,
            "format": response_format,
        }
        if config.stream_audio and response_format == "pcm":
            start_payload["sample_rate"] = _PCM_SAMPLE_RATE
        await websocket.send_json(start_payload)

        total_bytes = 0
        generation_failed = False
        request_id = None
        try:
            if config.stream_audio:
                request_id, generator, _ = await self._speech_service._prepare_speech_generation(request)
                async with aclosing(self._speech_service._generate_pcm_chunks(generator, request_id)) as stream:
                    async for chunk in stream:
                        total_bytes += len(chunk)
                        await websocket.send_bytes(chunk)
            else:
                audio_bytes, _ = await self._speech_service._generate_audio_bytes(request)
                total_bytes = len(audio_bytes)
                await websocket.send_bytes(audio_bytes)
        except WebSocketDisconnect:
            if request_id is not None:
                try:
                    await self._speech_service.engine_client.abort(request_id)
                except Exception:
                    logger.debug("Failed to abort streaming speech request %s", request_id, exc_info=True)
            raise
        except Exception as e:
            generation_failed = True
            logger.error("Generation failed for sentence %d: %s", sentence_index, e)
            await self._send_error(websocket, f"Generation failed for sentence {sentence_index}: {e}")
        finally:
            try:
                await websocket.send_json(
                    {
                        "type": "audio.done",
                        "sentence_index": sentence_index,
                        "total_bytes": total_bytes,
                        "error": generation_failed,
                    }
                )
            except Exception:
                logger.debug("Failed to send audio.done for sentence %d", sentence_index, exc_info=True)

    # ------------------------------------------------------------------
    # Voice management
    # ------------------------------------------------------------------

    async def _handle_voice_message(self, websocket: WebSocket, msg: dict) -> None:
        """Dispatch a voice.* message to the appropriate handler."""
        msg_type = msg["type"]
        if msg_type == "voice.list":
            await self._handle_voice_list(websocket)
        elif msg_type == "voice.add":
            await self._handle_voice_add(websocket, msg)
        elif msg_type == "voice.remove":
            await self._handle_voice_remove(websocket, msg)

    async def _handle_voice_list(self, websocket: WebSocket) -> None:
        """Return all available voices (built-in + uploaded)."""
        svc = self._speech_service
        voices = sorted(svc.supported_speakers) if svc.supported_speakers else []

        uploaded_voices: list[dict] = []
        for voice_name, info in getattr(svc, "uploaded_speakers", {}).items():
            entry: dict = {
                "name": info.get("name", voice_name),
                "consent": info.get("consent", ""),
                "created_at": info.get("created_at", 0),
                "file_size": info.get("file_size", 0),
                "mime_type": info.get("mime_type", ""),
                "embedding_source": info.get("embedding_source", "audio"),
                "embedding_dim": info.get("embedding_dim"),
            }
            if info.get("ref_text"):
                entry["ref_text"] = info["ref_text"]
            if info.get("speaker_description"):
                entry["speaker_description"] = info["speaker_description"]
            uploaded_voices.append(entry)

        await websocket.send_json({
            "type": "voice.list.result",
            "voices": voices,
            "uploaded_voices": uploaded_voices,
        })

    async def _handle_voice_add(self, websocket: WebSocket, msg: dict) -> None:
        """Add a voice from base64 audio or a speaker embedding vector."""
        name = msg.get("name")
        consent = msg.get("consent")
        if not name or not isinstance(name, str):
            await self._send_error(websocket, "voice.add requires a 'name' string")
            return
        if not consent or not isinstance(consent, str):
            await self._send_error(websocket, "voice.add requires a 'consent' string")
            return

        speaker_embedding = msg.get("speaker_embedding")
        audio_data = msg.get("audio_data")

        if speaker_embedding is not None and audio_data is not None:
            await self._send_error(
                websocket,
                "voice.add: 'audio_data' and 'speaker_embedding' are mutually exclusive",
            )
            return
        if speaker_embedding is None and audio_data is None:
            await self._send_error(
                websocket,
                "voice.add requires either 'audio_data' (base64) or 'speaker_embedding'",
            )
            return

        svc = self._speech_service
        try:
            if speaker_embedding is not None:
                embedding_json = (
                    json.dumps(speaker_embedding)
                    if isinstance(speaker_embedding, list)
                    else speaker_embedding
                )
                result = await svc.upload_voice_embedding(embedding_json, consent, name)
            else:
                audio_bytes = base64.b64decode(audio_data)
                audio_format = msg.get("audio_format", "wav")
                content_type = _MIME_FROM_FORMAT.get(audio_format, "application/octet-stream")
                upload_file = UploadFile(
                    file=io.BytesIO(audio_bytes),
                    size=len(audio_bytes),
                    filename=f"{name}.{audio_format}",
                    headers=Headers({"content-type": content_type}),
                )
                result = await svc.upload_voice(
                    upload_file,
                    consent,
                    name,
                    ref_text=msg.get("ref_text"),
                    speaker_description=msg.get("speaker_description"),
                )

            await websocket.send_json({
                "type": "voice.add.result",
                "success": True,
                "voice": result,
            })
        except ValueError as exc:
            await websocket.send_json({
                "type": "voice.add.result",
                "success": False,
                "error": str(exc),
            })
        except Exception as exc:
            logger.error("voice.add failed for '%s': %s", name, exc)
            await websocket.send_json({
                "type": "voice.add.result",
                "success": False,
                "error": f"Failed to add voice: {exc}",
            })

    async def _handle_voice_remove(self, websocket: WebSocket, msg: dict) -> None:
        """Remove an uploaded voice by name."""
        name = msg.get("name")
        if not name or not isinstance(name, str):
            await self._send_error(websocket, "voice.remove requires a 'name' string")
            return

        try:
            success = await self._speech_service.delete_voice(name)
            payload: dict = {
                "type": "voice.remove.result",
                "success": success,
                "name": name,
            }
            if not success:
                payload["error"] = f"Voice '{name}' not found"
            await websocket.send_json(payload)
        except Exception as exc:
            logger.error("voice.remove failed for '%s': %s", name, exc)
            await websocket.send_json({
                "type": "voice.remove.result",
                "success": False,
                "name": name,
                "error": str(exc),
            })

    # ------------------------------------------------------------------

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": message,
                }
            )
        except Exception:
            pass  # Connection may already be closed; safe to ignore
