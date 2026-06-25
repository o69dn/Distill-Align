"""Property-based tests for schemas."""

from distill_align.core.schemas import DataChunk, SourceMetadata
from hypothesis import given
from hypothesis import strategies as st


class TestBasic:
    @given(content=st.text(min_size=1, max_size=100))
    def test_roundtrip(self, content):
        meta = SourceMetadata(file_path="/test/f.md", file_name="f.md")
        chunk = DataChunk(content=content, metadata=meta)
        restored = DataChunk(**chunk.model_dump())
        assert restored.content == content


st_roles = st.sampled_from(["system", "user", "assistant", "tool"])
st_short_text = st.text(min_size=1, max_size=100)
st_medium_text = st.text(min_size=1, max_size=500)
st_file_path = st.text(min_size=5, max_size=80).map(lambda s: "/test/" + s.replace(" ", "_") + ".md")
st_file_name = st.text(min_size=3, max_size=30).map(lambda s: s.replace(" ", "_") + ".md")


class TestDataChunkProperties:
    """Property-based tests for DataChunk."""

    @given(content=st_medium_text, path=st_file_path, name=st_file_name)
    def test_content_roundtrip(self, content, path, name):
        meta = SourceMetadata(file_path=path, file_name=name)
        chunk = DataChunk(content=content, metadata=meta)
        restored = DataChunk(**chunk.model_dump())
        assert restored.content == content
        assert restored.metadata.file_path == path

    @given(content=st_medium_text, path=st_file_path, name=st_file_name)
    def test_auto_id_deterministic(self, content, path, name):
        meta = SourceMetadata(file_path=path, file_name=name)
        c1 = DataChunk(content=content, metadata=meta)
        c2 = DataChunk(content=content, metadata=meta)
        assert c1.id == c2.id
        assert len(c1.id) == 16


class TestConversationProperties:
    """Property-based tests for ConversationSchema."""

    @given(st_medium_text, st_medium_text)
    def test_user_assistant_roundtrip(self, user_msg, asst_msg):
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        conv = ConversationSchema(
            id="prop-id",
            source_chunk_id="chunk-prop",
            turns=[
                SynthesizedTurn(role="user", content=user_msg),
                SynthesizedTurn(role="assistant", content=asst_msg),
            ],
        )
        restored = ConversationSchema(**conv.model_dump())
        assert restored.get_user_turns()[0].content == user_msg
        assert restored.get_assistant_turns()[0].content == asst_msg

    @given(content=st_short_text)
    def test_system_prompt_preserved(self, content):
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        conv = ConversationSchema(
            id="sys-id",
            source_chunk_id="chunk-sys",
            turns=[
                SynthesizedTurn(role="system", content=content),
                SynthesizedTurn(role="user", content="Hello"),
                SynthesizedTurn(role="assistant", content="Hi"),
            ],
        )
        restored = ConversationSchema(**conv.model_dump())
        assert restored.get_system_prompt() == content


class TestConfigProperties:
    """Property-based tests for config schemas."""

    @given(chunk_size=st.integers(min_value=1, max_value=10000))
    def test_ingestion_config_chunk_size(self, chunk_size):
        from distill_align.core.schemas import IngestionConfig

        config = IngestionConfig(chunk_size=chunk_size)
        assert config.chunk_size == chunk_size

    @given(concurrency=st.integers(min_value=1, max_value=50))
    def test_synthesis_config_concurrency(self, concurrency):
        from distill_align.core.schemas import SynthesisConfig

        config = SynthesisConfig(max_concurrency=concurrency)
        assert config.max_concurrency == concurrency

    @given(temperature=st.floats(min_value=0.0, max_value=2.0))
    def test_synthesis_config_temperature(self, temperature):
        from distill_align.core.schemas import SynthesisConfig

        config = SynthesisConfig(temperature=temperature)
        assert 0.0 <= config.temperature <= 2.0

    @given(formats=st.lists(st.sampled_from(["sharegpt", "alpaca"]), min_size=1, max_size=2))
    def test_export_config_formats(self, formats):
        from distill_align.core.schemas import ExportConfig

        config = ExportConfig(formats=formats)
        assert set(formats).issubset(config.formats)


class TestAlpacaEntryProperties:
    """Property-based tests for AlpacaEntry."""

    @given(instruction=st_short_text, output=st_short_text)
    def test_valid_entry(self, instruction, output):
        from distill_align.core.schemas import AlpacaEntry

        entry = AlpacaEntry(instruction=instruction, input="", output=output)
        assert entry.instruction == instruction
        assert entry.output == output


class TestShareGPTMessageProperties:
    """Property-based tests for ShareGPTMessage."""

    @given(value=st_short_text)
    def test_message_from_human(self, value):
        from distill_align.core.schemas import ShareGPTMessage

        msg = ShareGPTMessage(from_="human", value=value)
        assert msg.value == value
        data = msg.model_dump(by_alias=True)
        assert data["from"] == "human"

    @given(value=st_short_text)
    def test_message_from_gpt(self, value):
        from distill_align.core.schemas import ShareGPTMessage

        msg = ShareGPTMessage(from_="gpt", value=value)
        assert msg.value == value
        assert msg.from_ == "gpt"
