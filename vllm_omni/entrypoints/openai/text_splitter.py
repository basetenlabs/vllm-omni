"""Multi-language sentence boundary detector for streaming TTS input.

Buffers incoming text and splits at sentence boundaries (English and CJK),
yielding complete sentences for audio generation.

Includes :class:`AsymmetricSplitter` for two-phase chunking that emits a
small "lead" chunk as soon as possible (low TTFA) and switches to larger
"steady" chunks afterwards for more stable prosody.
"""

import re
from dataclasses import dataclass
from re import Pattern

# Maximum buffer size (in characters) to prevent unbounded memory growth.
_MAX_BUFFER_SIZE = 100_000  # ~100 KB of text

# Sentence-level: .!? + CJK sentence-ending 。！？
# NOTE: English requires trailing whitespace to confirm a boundary —
# end-of-string is NOT treated as a boundary (that is what flush() is for).
SPLIT_SENTENCE = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=[。！？])"
)

# Clause-level: adds CJK commas ， and semicolons ；
SPLIT_CLAUSE = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=[。！？，；])"
)

# Default alias
_SENTENCE_BOUNDARY_RE = SPLIT_SENTENCE


class SentenceSplitter:
    """Incremental sentence splitter for streaming text input.

    Buffers text and yields complete sentences when boundaries are detected.
    Designed for TTS pipelines where text arrives incrementally (e.g., from STT).

    Args:
        min_sentence_length: Minimum character length for a sentence.
            Sentences shorter than this are kept in the buffer to avoid
            splitting on abbreviations like "Dr." or "U.S.".
        boundary_re: Custom compiled regex for sentence boundaries.
            Use ``SPLIT_SENTENCE`` (default) for sentence-level splitting,
            ``SPLIT_CLAUSE`` for finer-grained clause-level splitting,
            or pass your own ``re.Pattern``.
    """

    def __init__(
        self,
        min_sentence_length: int = 2,
        boundary_re: Pattern[str] | None = None,
    ) -> None:
        self._buffer: str = ""
        self._min_sentence_length = min_sentence_length
        self._boundary_re = boundary_re or _SENTENCE_BOUNDARY_RE

    @property
    def buffer(self) -> str:
        """Current buffered text."""
        return self._buffer

    def add_text(self, text: str) -> list[str]:
        """Add text to the buffer and return any complete sentences.

        Args:
            text: Incoming text chunk.

        Returns:
            List of complete sentences extracted from the buffer.
            May be empty if no sentence boundary was found.

        Raises:
            ValueError: If the buffer exceeds the maximum size.
        """
        if not text:
            return []

        self._buffer += text
        if len(self._buffer) > _MAX_BUFFER_SIZE:
            raise ValueError(
                f"Text buffer exceeded maximum size ({_MAX_BUFFER_SIZE} chars). "
                "Consider adding sentence-ending punctuation to your input."
            )
        return self._extract_sentences()

    def flush(self) -> str | None:
        """Flush remaining buffered text as a final sentence.

        Returns:
            The remaining buffered text (stripped), or None if buffer is empty.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None

    def _extract_sentences(self) -> list[str]:
        """Split buffer at sentence boundaries, keeping incomplete text buffered."""
        parts = self._boundary_re.split(self._buffer)

        if len(parts) <= 1:
            # No boundary found — keep everything in buffer
            return []

        sentences: list[str] = []
        carry = ""
        # All parts except the last are complete sentences
        for i in range(len(parts) - 1):
            text = carry + parts[i]
            carry = ""
            stripped = text.strip()
            if len(stripped) >= self._min_sentence_length:
                sentences.append(stripped)
            elif stripped:
                # Too short (e.g. "Dr.") — carry forward to next part
                carry = text
            # else: empty, skip

        # Last part stays in buffer (may be incomplete)
        self._buffer = carry + parts[-1]

        return sentences


@dataclass
class StreamChunk:
    """A piece of buffered text ready for synthesis.

    Attributes:
        text: The chunk content (already stripped of leading/trailing whitespace).
        is_lead: ``True`` only for the first chunk in a session — used by
            the handler to apply lead-only optimizations (e.g. a smaller
            ``initial_codec_chunk_frames`` override) without affecting later
            chunks.
    """

    text: str
    is_lead: bool


class AsymmetricSplitter:
    """Two-phase splitter for low-TTFA streaming TTS.

    Phase 1 — *lead* (one chunk per session):
        Emit the **first** synthesizable unit as early as possible. Tries,
        in order:

        1. Sentence boundary anywhere in the buffer.
        2. ``lead_boundary_re`` (default: clause) once the buffer has at
           least ``lead_min_chars`` characters.
        3. Whitespace-cut once the buffer has at least ``lead_min_chars``
           characters and contains a space.
        4. :meth:`force_lead_flush` — caller-driven timer fallback that
           emits whatever is buffered (cut at the last whitespace if any).

    Phase 2 — *steady* (everything after the lead):
        Accumulate complete sentences using ``steady_boundary_re`` (default:
        sentence). Emit chunks of ``steady_units_per_chunk`` sentences at a
        time. Optional paragraph-break fast-path (``\\n\\n``) flushes any
        accumulated sentences plus the paragraph head together.

    The lead policy trades a tiny amount of prosody on the very first
    utterance for a much faster time-to-first-audio. The steady policy
    keeps later audio coherent by synthesizing larger spans.
    """

    def __init__(
        self,
        lead_min_chars: int = 32,
        lead_boundary_re: Pattern[str] | None = None,
        steady_boundary_re: Pattern[str] | None = None,
        steady_units_per_chunk: int = 1,
        steady_paragraph_break: bool = True,
        min_sentence_length: int = 2,
    ) -> None:
        if lead_min_chars < 1:
            raise ValueError("lead_min_chars must be >= 1")
        if steady_units_per_chunk < 1:
            raise ValueError("steady_units_per_chunk must be >= 1")

        self._buffer: str = ""
        self._lead_emitted: bool = False
        self._lead_min_chars = lead_min_chars
        self._lead_boundary_re = lead_boundary_re or SPLIT_CLAUSE
        self._steady_boundary_re = steady_boundary_re or SPLIT_SENTENCE
        self._steady_units_per_chunk = steady_units_per_chunk
        self._steady_paragraph_break = steady_paragraph_break
        self._min_sentence_length = min_sentence_length
        # Holds complete sentences awaiting grouping into a steady chunk.
        self._steady_carry: list[str] = []

    @property
    def buffer(self) -> str:
        """Current unflushed text (lead phase) or partial sentence (steady phase)."""
        return self._buffer

    @property
    def lead_emitted(self) -> bool:
        """True once the lead chunk has been emitted by any path."""
        return self._lead_emitted

    @property
    def has_pending_text(self) -> bool:
        """True if there is buffered text or carried sentences not yet emitted."""
        return bool(self._buffer.strip()) or bool(self._steady_carry)

    def add_text(self, text: str) -> list[StreamChunk]:
        """Append text and return any complete chunks ready for synthesis."""
        if not text:
            return []
        self._buffer += text
        if len(self._buffer) > _MAX_BUFFER_SIZE:
            raise ValueError(
                f"Text buffer exceeded maximum size ({_MAX_BUFFER_SIZE} chars). "
                "Consider adding sentence-ending punctuation to your input."
            )

        chunks: list[StreamChunk] = []
        if not self._lead_emitted:
            lead = self._try_extract_lead()
            if lead is not None:
                chunks.append(StreamChunk(text=lead, is_lead=True))
                self._lead_emitted = True

        if self._lead_emitted:
            chunks.extend(self._extract_steady())
        return chunks

    def force_lead_flush(self) -> StreamChunk | None:
        """Caller-driven timer fallback for the lead chunk.

        Emits the current buffer (cut at the last whitespace if present)
        as the lead even when no preferred boundary has been found. Has
        no effect once the lead has already been emitted.
        """
        if self._lead_emitted:
            return None
        text = self._cut_lead_force()
        if text is None:
            return None
        self._lead_emitted = True
        return StreamChunk(text=text, is_lead=True)

    def flush(self) -> StreamChunk | None:
        """End-of-input flush.

        Returns any remaining carried sentences plus the buffer as one
        final chunk. If the lead was never emitted, the chunk is marked
        as lead.
        """
        parts: list[str] = []
        if self._steady_carry:
            parts.append(" ".join(self._steady_carry).strip())
            self._steady_carry = []
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            parts.append(remaining)

        joined = " ".join(p for p in parts if p).strip()
        if not joined:
            return None
        is_lead = not self._lead_emitted
        if is_lead:
            self._lead_emitted = True
        return StreamChunk(text=joined, is_lead=is_lead)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_extract_lead(self) -> str | None:
        """Try increasingly aggressive cuts for the lead chunk.

        Order:
            1. Below ``lead_min_chars``: only a sentence boundary may cut.
            2. At/above ``lead_min_chars``: cut at the **earliest** boundary
               among sentence and ``lead_boundary_re`` (clause-level
               regexes typically include sentence punctuation already, so
               this naturally picks the earliest viable cut).
            3. Otherwise, fall back to the last whitespace once the buffer
               is past ``lead_min_chars`` so we don't slice mid-word.
        """
        sent_match = SPLIT_SENTENCE.search(self._buffer)
        sent_end = sent_match.end() if sent_match else None

        if len(self._buffer) < self._lead_min_chars:
            if sent_end is not None:
                return self._cut_lead_at(sent_end)
            return None

        lead_match = self._lead_boundary_re.search(self._buffer)
        lead_end = lead_match.end() if lead_match else None

        candidates = [p for p in (sent_end, lead_end) if p is not None]
        if candidates:
            return self._cut_lead_at(min(candidates))

        idx = self._buffer.rfind(" ")
        if idx >= self._lead_min_chars - 1:
            return self._cut_lead_at(idx + 1)
        return None

    def _cut_lead_at(self, cut: int) -> str | None:
        text = self._buffer[:cut].strip()
        if not text:
            return None
        self._buffer = self._buffer[cut:]
        return text

    def _cut_lead_force(self) -> str | None:
        """Fallback emit: prefer last whitespace, else flush whole buffer."""
        if not self._buffer.strip():
            return None
        idx = self._buffer.rfind(" ")
        if idx > 0:
            text = self._buffer[:idx].strip()
            self._buffer = self._buffer[idx + 1 :]
        else:
            text = self._buffer.strip()
            self._buffer = ""
        return text or None

    def _extract_steady(self) -> list[StreamChunk]:
        """Extract grouped steady-phase chunks from the buffer."""
        out: list[StreamChunk] = []

        if self._steady_paragraph_break and "\n\n" in self._buffer:
            head, _, tail = self._buffer.partition("\n\n")
            head_stripped = head.strip()
            pieces = list(self._steady_carry)
            if head_stripped:
                pieces.append(head_stripped)
            self._steady_carry = []
            self._buffer = tail
            joined = " ".join(p for p in pieces if p).strip()
            if joined:
                out.append(StreamChunk(text=joined, is_lead=False))

        parts = self._steady_boundary_re.split(self._buffer)
        if len(parts) <= 1:
            return out

        carry = ""
        complete: list[str] = []
        for i in range(len(parts) - 1):
            text = (carry + parts[i]).strip()
            carry = ""
            if len(text) >= self._min_sentence_length:
                complete.append(text)
            elif text:
                # Too short (e.g. abbreviation) — carry forward.
                carry = text + " "
        self._buffer = carry + parts[-1]

        for sentence in complete:
            self._steady_carry.append(sentence)
            if len(self._steady_carry) >= self._steady_units_per_chunk:
                joined = " ".join(self._steady_carry).strip()
                self._steady_carry = []
                if joined:
                    out.append(StreamChunk(text=joined, is_lead=False))
        return out
