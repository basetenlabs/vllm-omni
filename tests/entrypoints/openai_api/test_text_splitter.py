"""Tests for SentenceSplitter used in streaming TTS input."""

import pytest

from vllm_omni.entrypoints.openai.text_splitter import (
    SPLIT_CLAUSE,
    SPLIT_SENTENCE,
    AsymmetricSplitter,
    SentenceSplitter,
    StreamChunk,
)

pytestmark = [pytest.mark.openai, pytest.mark.speech, pytest.mark.core_model, pytest.mark.cpu]


class TestSentenceSplitterEnglish:
    """Tests for English sentence splitting."""

    def test_single_sentence_no_boundary(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Hello world")
        assert result == []
        assert splitter.buffer == "Hello world"

    def test_single_sentence_with_boundary(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Hello world. How are you?")
        assert len(result) == 1
        assert result[0] == "Hello world."

    def test_multiple_sentences(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Hello. How are you? I am fine! ")
        assert len(result) == 3
        assert result[0] == "Hello."
        assert result[1] == "How are you?"
        assert result[2] == "I am fine!"

    def test_exclamation_mark(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Wow, that is great! Tell me more.")
        assert len(result) == 1
        assert result[0] == "Wow, that is great!"

    def test_question_mark(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Can you hear me? I hope so.")
        assert len(result) == 1
        assert result[0] == "Can you hear me?"


class TestSentenceSplitterChinese:
    """Tests for CJK sentence splitting."""

    def test_chinese_period(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("你好世界。你好吗")
        assert len(result) == 1
        assert result[0] == "你好世界。"

    def test_chinese_exclamation(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("太好了！谢谢你")
        assert len(result) == 1
        assert result[0] == "太好了！"

    def test_chinese_question(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("你是谁？我是小明")
        assert len(result) == 1
        assert result[0] == "你是谁？"

    def test_chinese_comma_no_split(self):
        """Chinese commas are clause-level and should not trigger a split."""
        splitter = SentenceSplitter()
        result = splitter.add_text("你好，世界")
        assert result == []
        assert splitter.buffer == "你好，世界"

    def test_chinese_semicolon_no_split(self):
        """Chinese semicolons are clause-level and should not trigger a split."""
        splitter = SentenceSplitter()
        result = splitter.add_text("第一点；第二点")
        assert result == []
        assert splitter.buffer == "第一点；第二点"

    def test_chinese_multiple(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("你好！你好吗？我很好。")
        assert len(result) == 3
        assert result[0] == "你好！"
        assert result[1] == "你好吗？"
        assert result[2] == "我很好。"


class TestSentenceSplitterMixed:
    """Tests for mixed-language sentence splitting."""

    def test_mixed_english_chinese(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Hello世界。How are you? ")
        assert len(result) == 2
        assert result[0] == "Hello世界。"
        assert result[1] == "How are you?"


class TestSentenceSplitterIncremental:
    """Tests for incremental (multi-chunk) text input."""

    def test_accumulation_across_chunks(self):
        splitter = SentenceSplitter()
        # First chunk: no boundary
        result1 = splitter.add_text("Hello ")
        assert result1 == []

        # Second chunk: completes a sentence
        result2 = splitter.add_text("world. How")
        assert len(result2) == 1
        assert result2[0] == "Hello world."
        assert splitter.buffer == "How"

    def test_word_by_word(self):
        splitter = SentenceSplitter()
        words = ["Hello, ", "how ", "are ", "you? ", "I ", "am ", "fine."]
        all_sentences = []
        for word in words:
            all_sentences.extend(splitter.add_text(word))

        assert len(all_sentences) == 1
        assert all_sentences[0] == "Hello, how are you?"
        # "I am fine." stays in buffer (no trailing whitespace after period)

    def test_three_chunks(self):
        splitter = SentenceSplitter()
        splitter.add_text("The quick brown ")
        splitter.add_text("fox jumps. ")
        result = splitter.add_text("Over the lazy dog. ")
        # "The quick brown fox jumps." should have been returned on second chunk
        # "Over the lazy dog." on third chunk
        assert len(result) == 1
        assert result[0] == "Over the lazy dog."


class TestSentenceSplitterFlush:
    """Tests for flush behavior."""

    def test_flush_returns_remaining(self):
        splitter = SentenceSplitter()
        splitter.add_text("Hello world")
        result = splitter.flush()
        assert result == "Hello world"
        assert splitter.buffer == ""

    def test_flush_empty_buffer(self):
        splitter = SentenceSplitter()
        result = splitter.flush()
        assert result is None

    def test_flush_after_sentence(self):
        splitter = SentenceSplitter()
        splitter.add_text("Hello world. Remaining text")
        result = splitter.flush()
        assert result == "Remaining text"

    def test_flush_whitespace_only(self):
        splitter = SentenceSplitter()
        splitter.add_text("Hello. ")
        # "Hello." extracted, buffer is " "
        result = splitter.flush()
        # Whitespace-only should return None
        assert result is None

    def test_flush_clears_buffer(self):
        splitter = SentenceSplitter()
        splitter.add_text("some text")
        splitter.flush()
        assert splitter.buffer == ""
        # Second flush should return None
        assert splitter.flush() is None


class TestSentenceSplitterEdgeCases:
    """Edge case tests."""

    def test_empty_input(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("")
        assert result == []
        assert splitter.buffer == ""

    def test_none_like_empty(self):
        """Empty string should not affect buffer."""
        splitter = SentenceSplitter()
        splitter.add_text("Hello")
        splitter.add_text("")
        assert splitter.buffer == "Hello"

    def test_only_punctuation(self):
        splitter = SentenceSplitter()
        result = splitter.add_text(". ")
        # "." is 1 char, below default min_sentence_length of 2
        # It will be carried forward
        assert result == []

    def test_min_sentence_length(self):
        splitter = SentenceSplitter(min_sentence_length=10)
        result = splitter.add_text("Hi. Hello world. ")
        # "Hi." is 3 chars (< 10), so it gets carried to "Hello world."
        assert len(result) == 1
        assert "Hi." in result[0]
        assert "Hello world." in result[0]

    def test_short_segments_are_carried_until_long_enough(self):
        splitter = SentenceSplitter(min_sentence_length=10)
        result = splitter.add_text("Hi. Ok. Hello there. ")
        assert result == ["Hi.Ok.Hello there."]
        assert splitter.buffer == ""

    def test_min_sentence_length_zero(self):
        splitter = SentenceSplitter(min_sentence_length=0)
        result = splitter.add_text("A. B. ")
        assert len(result) == 2

    def test_no_boundary_then_flush(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Hello world how are you")
        assert result == []
        flushed = splitter.flush()
        assert flushed == "Hello world how are you"

    def test_consecutive_punctuation(self):
        splitter = SentenceSplitter()
        result = splitter.add_text("Really?! Yes, really. ")
        assert len(result) >= 1

    def test_reuse_after_flush(self):
        """Splitter can be reused after flush."""
        splitter = SentenceSplitter()
        splitter.add_text("First session.")
        splitter.flush()

        result = splitter.add_text("Second session. More text")
        assert len(result) == 1
        assert result[0] == "Second session."
        assert splitter.buffer == "More text"


class TestSentenceSplitterBufferLimit:
    """Tests for buffer overflow protection."""

    def test_buffer_overflow_raises(self):
        from vllm_omni.entrypoints.openai.text_splitter import _MAX_BUFFER_SIZE

        splitter = SentenceSplitter()
        # Fill buffer just under the limit
        splitter.add_text("x" * (_MAX_BUFFER_SIZE - 1))
        # One more char should exceed the limit
        with pytest.raises(ValueError, match="exceeded maximum size"):
            splitter.add_text("xx")


class TestAsymmetricSplitterLeadPhase:
    """Lead phase emits the very first synthesizable unit ASAP."""

    def test_lead_emits_on_sentence_boundary_below_min_chars(self):
        # Sentence boundaries always win, regardless of lead_min_chars.
        sp = AsymmetricSplitter(lead_min_chars=100)
        chunks = sp.add_text("Hi. ")
        assert len(chunks) == 1
        assert chunks[0] == StreamChunk(text="Hi.", is_lead=True)
        assert sp.lead_emitted is True

    def test_lead_waits_for_min_chars_with_clause(self):
        sp = AsymmetricSplitter(lead_min_chars=20)
        # Below min_chars and only a clause-style break: keep buffering.
        assert sp.add_text("Well, ") == []
        assert sp.lead_emitted is False
        # Cross min_chars and now have a clause boundary.
        chunks = sp.add_text("now we are getting somewhere, ")
        assert chunks and chunks[0].is_lead
        assert chunks[0].text.startswith("Well, ")

    def test_lead_falls_back_to_whitespace_after_min_chars(self):
        sp = AsymmetricSplitter(lead_min_chars=20)
        # No punctuation at all but enough chars + a whitespace.
        chunks = sp.add_text("aaaaa bbbbb ccccc ddddd eeeee fffff")
        assert chunks
        assert chunks[0].is_lead
        # Should cut at a whitespace, leaving the trailing word in buffer.
        assert sp.buffer.strip() != ""

    def test_force_lead_flush_emits_buffered_text(self):
        sp = AsymmetricSplitter(lead_min_chars=100)
        sp.add_text("the LLM has only emitted a fragment so far")
        forced = sp.force_lead_flush()
        assert forced is not None and forced.is_lead
        assert "fragment" in forced.text or "fragment" in (sp.buffer + forced.text)

    def test_force_lead_flush_noop_after_lead_emitted(self):
        sp = AsymmetricSplitter()
        sp.add_text("Hello world. ")
        assert sp.lead_emitted
        assert sp.force_lead_flush() is None


class TestAsymmetricSplitterSteadyPhase:
    """Steady phase emits grouped sentences after the lead is out."""

    def test_steady_groups_sentences(self):
        sp = AsymmetricSplitter(steady_units_per_chunk=2)
        # Lead first.
        lead_chunks = sp.add_text("Hi. ")
        assert lead_chunks[0].is_lead
        # Now feed three sentences. Steady should emit one chunk of 2,
        # carry one for next.
        chunks = sp.add_text("One. Two. Three. ")
        assert [c.is_lead for c in chunks] == [False]
        assert chunks[0].text == "One. Two."
        # Force the carry out via flush.
        tail = sp.flush()
        assert tail is not None and not tail.is_lead
        assert tail.text == "Three."

    def test_steady_paragraph_break_flushes_carry(self):
        sp = AsymmetricSplitter(steady_units_per_chunk=5)
        # Lead.
        sp.add_text("Hi. ")
        # Two sentences that would be carried since 5 > 2.
        chunks = sp.add_text("First. Second.\n\n")
        # Paragraph break emits everything seen so far as a steady chunk.
        steady = [c for c in chunks if not c.is_lead]
        assert steady, f"expected steady chunk on paragraph break, got {chunks}"
        joined = " ".join(c.text for c in steady)
        assert "First." in joined and "Second." in joined

    def test_lead_picks_earliest_boundary_cjk(self):
        # CJK commas ， are clause boundaries in SPLIT_CLAUSE. With min_chars
        # met, the lead should cut at the earliest comma, before the period.
        sp = AsymmetricSplitter(
            lead_min_chars=4,
            lead_boundary_re=SPLIT_CLAUSE,
        )
        chunks = sp.add_text("从前，有一个国王。")
        assert chunks[0].is_lead
        assert chunks[0].text.endswith("，")
        # Remaining sentence should be available after the lead.
        more = sp.add_text("")
        # The steady phase needs a flush to drain since no more boundaries arrive.
        tail = sp.flush()
        assert tail is None or "国王。" in tail.text or "国王" in tail.text

    def test_lead_below_min_chars_only_sentence_boundary_fires(self):
        sp = AsymmetricSplitter(lead_min_chars=50)
        # CJK comma alone won't fire below min_chars.
        assert sp.add_text("从前，") == []
        # Sentence-final does.
        chunks = sp.add_text("有一个国王。")
        assert chunks and chunks[0].is_lead
        assert "国王" in chunks[0].text


class TestAsymmetricSplitterFlushAndEmpty:
    def test_flush_lead_when_no_text_seen(self):
        sp = AsymmetricSplitter()
        assert sp.flush() is None

    def test_flush_marks_remaining_as_lead_if_never_emitted(self):
        sp = AsymmetricSplitter()
        sp.add_text("only a tiny fragment with no boundary")
        out = sp.flush()
        assert out is not None
        assert out.is_lead is True
        assert "fragment" in out.text

    def test_empty_text_does_not_advance_state(self):
        sp = AsymmetricSplitter()
        assert sp.add_text("") == []
        assert sp.lead_emitted is False
        assert sp.has_pending_text is False

    def test_invalid_init_raises(self):
        with pytest.raises(ValueError):
            AsymmetricSplitter(lead_min_chars=0)
        with pytest.raises(ValueError):
            AsymmetricSplitter(steady_units_per_chunk=0)
