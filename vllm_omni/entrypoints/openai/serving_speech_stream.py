"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence using the existing TTS pipeline.

Also supports voice management (list, upload, delete) so clients can manage
voice clones over the same WebSocket connection used for TTS.

Protocol:
    Client -> Server (voice management — can be sent at any time):
        {"type": "voice.list"}
        {"type": "voice.upload", "name": "...", "consent": "...",
         "audio_data": "<base64>", "mime_type": "audio/wav",
         "filename": "sample.wav", "ref_text": "..."}
        {"type": "voice.upload", "name": "...", "consent": "...",
         "speaker_embedding": [0.1, ...]}
        {"type": "voice.delete", "name": "..."}

    Client -> Server (TTS streaming):
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input

    Server -> Client (voice management responses):
        {"type": "voice.list.result", "voices": [...], "uploaded_voices": [...]}
        {"type": "voice.upload.result", "success": true, "voice": {...}}
        {"type": "voice.delete.result", "success": true, "name": "..."}

    Server -> Client (TTS streaming responses):
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        {"type": "audio.done", "sentence_index": 0}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}
"""

import asyncio
import base64
import io
import json
from contextlib import aclosing

from fastapi import WebSocket, WebSocketDisconnect, UploadFile
from starlette.datastructures import Headers
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

    _VOICE_MSG_TYPES = {"voice.list", "voice.upload", "voice.delete"}

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection.

        Voice management messages (voice.list, voice.upload, voice.delete) are
        accepted at any point during the session — before session.config, during
        text streaming, or standalone without any TTS at all.
        """
        await websocket.accept()

        try:
            config: StreamingSpeechSessionConfig | None = None
            splitter: SentenceSplitter | None = None
            sentence_index = 0
            has_received_message = False

            while True:
                # Use large message size limit before config (voice uploads can be large)
                max_size = _MAX_CONFIG_MESSAGE_SIZE if config is None else _MAX_INPUT_TEXT_MESSAGE_SIZE
                # First message gets the short config timeout; once any message
                # has been received, use the longer idle timeout.
                timeout = self._config_timeout if not has_received_message else self._idle_timeout

                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    await self._send_error(websocket, "Idle timeout: no message received")
                    return

                if len(raw) > max_size:
                    await self._send_error(websocket, "Message too large")
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                    continue

                if not isinstance(msg, dict):
                    await self._send_error(websocket, "WebSocket messages must be JSON objects")
                    continue

                msg_type = msg.get("type")
                has_received_message = True

                # --- Voice management (accepted at any time) ---
                if msg_type in self._VOICE_MSG_TYPES:
                    await self._handle_voice_message(websocket, msg_type, msg)
                    continue

                # --- Session config ---
                if msg_type == "session.config":
                    if config is not None:
                        await self._send_error(websocket, "session.config already received")
                        continue
                    config = self._parse_config(msg)
                    if config is None:
                        await self._send_error(websocket, "Invalid session.config")
                        return

                    if config.model and hasattr(self._speech_service, "_check_model"):
                        error = await self._speech_service._check_model(
                            OpenAICreateSpeechRequest(input="ping", model=config.model)
                        )
                        if error is not None:
                            await self._send_error(websocket, str(error))
                            return

                    boundary_re = SPLIT_CLAUSE if config.split_granularity == "clause" else SPLIT_SENTENCE
                    splitter = SentenceSplitter(boundary_re=boundary_re)
                    continue

                # --- TTS text streaming (requires session.config first) ---
                if msg_type == "input.text":
                    if config is None or splitter is None:
                        await self._send_error(websocket, "Send session.config before input.text")
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
                    if config is None or splitter is None:
                        await self._send_error(websocket, "Send session.config before input.done")
                        continue
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

    @staticmethod
    def _parse_config(msg: dict) -> StreamingSpeechSessionConfig | None:
        """Parse a session.config message dict into a validated config object."""
        try:
            return StreamingSpeechSessionConfig(**{k: v for k, v in msg.items() if k != "type"})
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Voice management handlers                                          #
    # ------------------------------------------------------------------ #

    async def _handle_voice_message(self, websocket: WebSocket, msg_type: str, msg: dict) -> None:
        """Dispatch a voice.* message to the appropriate handler."""
        try:
            if msg_type == "voice.list":
                await self._handle_voice_list(websocket)
            elif msg_type == "voice.upload":
                await self._handle_voice_upload(websocket, msg)
            elif msg_type == "voice.delete":
                await self._handle_voice_delete(websocket, msg)
        except Exception as e:
            logger.exception("Voice management error (%s): %s", msg_type, e)
            await self._send_error(websocket, f"Voice management error: {e}")

    async def _handle_voice_list(self, websocket: WebSocket) -> None:
        svc = self._speech_service
        voices = sorted(svc.supported_speakers) if svc.supported_speakers else []
        uploaded_voices = []
        for voice_name, info in svc.uploaded_speakers.items():
            uploaded_voices.append({
                "name": info.get("name", voice_name),
                "consent": info.get("consent", ""),
                "created_at": info.get("created_at", 0),
                "file_size": info.get("file_size", 0),
                "mime_type": info.get("mime_type", ""),
                "embedding_source": info.get("embedding_source", "audio"),
                "embedding_dim": info.get("embedding_dim"),
            })
        await websocket.send_json({
            "type": "voice.list.result",
            "voices": voices,
            "uploaded_voices": uploaded_voices,
        })

    async def _handle_voice_upload(self, websocket: WebSocket, msg: dict) -> None:
        name = msg.get("name")
        consent = msg.get("consent")
        if not name or not consent:
            await self._send_error(websocket, "voice.upload requires 'name' and 'consent'")
            return

        speaker_embedding = msg.get("speaker_embedding")
        audio_data_b64 = msg.get("audio_data")

        if speaker_embedding is not None and audio_data_b64 is not None:
            await self._send_error(
                websocket, "'audio_data' and 'speaker_embedding' are mutually exclusive"
            )
            return

        svc = self._speech_service

        if speaker_embedding is not None:
            embedding_json = json.dumps(speaker_embedding)
            result = await svc.upload_voice_embedding(embedding_json, consent, name)
        elif audio_data_b64 is not None:
            if not isinstance(audio_data_b64, str):
                await self._send_error(websocket, "'audio_data' must be a base64-encoded string")
                return
            try:
                audio_bytes = base64.b64decode(audio_data_b64)
            except Exception:
                await self._send_error(websocket, "Invalid base64 in 'audio_data'")
                return

            mime_type = msg.get("mime_type", "audio/wav")
            filename = msg.get("filename", "upload.wav")
            ref_text = msg.get("ref_text")

            upload_file = UploadFile(
                file=io.BytesIO(audio_bytes),
                filename=filename,
                headers=Headers({"content-type": mime_type}),
            )
            result = await svc.upload_voice(upload_file, consent, name, ref_text=ref_text)
        else:
            await self._send_error(
                websocket, "voice.upload requires 'audio_data' (base64) or 'speaker_embedding'"
            )
            return

        await websocket.send_json({
            "type": "voice.upload.result",
            "success": True,
            "voice": result,
        })

    async def _handle_voice_delete(self, websocket: WebSocket, msg: dict) -> None:
        name = msg.get("name")
        if not name:
            await self._send_error(websocket, "voice.delete requires 'name'")
            return

        svc = self._speech_service
        success = await svc.delete_voice(name)
        if not success:
            await websocket.send_json({
                "type": "voice.delete.result",
                "success": False,
                "name": name,
                "error": f"Voice '{name}' not found",
            })
            return

        await websocket.send_json({
            "type": "voice.delete.result",
            "success": True,
            "name": name,
        })

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
                async with aclosing(self._speech_service._generate_audio_chunks(generator, request_id)) as stream:
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
