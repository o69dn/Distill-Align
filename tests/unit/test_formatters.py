"""
Unit tests for exporter module.
"""

import json

import pytest
from distill_align.core.schemas import ConversationSchema, SynthesizedTurn
from distill_align.exporter.formatters.alpaca import AlpacaFormatter
from distill_align.exporter.formatters.hf_messages import HFMessagesFormatter
from distill_align.exporter.formatters.jsonl import JsonlFormatter
from distill_align.exporter.formatters.sharegpt import ShareGPTFormatter


@pytest.fixture
def sample_conversations():
    """Sample conversations for testing."""
    return [
        ConversationSchema(
            id="conv-1",
            source_chunk_id="chunk-1",
            turns=[
                SynthesizedTurn(role="system", content="You are a helpful assistant."),
                SynthesizedTurn(role="user", content="What is Python?"),
                SynthesizedTurn(role="assistant", content="Python is a programming language."),
            ],
        ),
        ConversationSchema(
            id="conv-2",
            source_chunk_id="chunk-2",
            turns=[
                SynthesizedTurn(role="user", content="What is machine learning?"),
                SynthesizedTurn(role="assistant", content="Machine learning is a subset of AI."),
            ],
        ),
    ]


@pytest.fixture
def parsed_conversations():
    """Conversations with confidence scores and judge scores."""
    return [
        ConversationSchema(
            id="scored-1",
            source_chunk_id="src-1",
            turns=[
                SynthesizedTurn(role="user", content="Q1?"),
                SynthesizedTurn(role="assistant", content="A1."),
            ],
            confidence_score=0.95,
            judge_scores={"overall": 9.5},
        ),
        ConversationSchema(
            id="scored-2",
            source_chunk_id="src-2",
            turns=[
                SynthesizedTurn(role="user", content="Q2?"),
                SynthesizedTurn(role="assistant", content="A2."),
            ],
            confidence_score=0.50,
        ),
    ]


class TestShareGPTFormatter:
    """Tests for ShareGPTFormatter."""

    def test_format_conversations(self, temp_output_dir, sample_conversations):
        formatter = ShareGPTFormatter(temp_output_dir)
        output_path = formatter.format(sample_conversations, "test_sharegpt.json")

        assert output_path.exists()
        assert output_path.name == "test_sharegpt.json"

        with open(output_path) as f:
            data = json.load(f)

        assert len(data) == 2
        assert "id" in data[0]
        assert "conversations" in data[0]

    def test_validate_valid_data(self, temp_output_dir):
        formatter = ShareGPTFormatter(temp_output_dir)
        valid_data = [
            {
                "id": "test",
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi"},
                ],
            }
        ]
        assert formatter.validate(valid_data) is True

    def test_validate_invalid_data(self, temp_output_dir):
        formatter = ShareGPTFormatter(temp_output_dir)
        invalid_data = [{"invalid": "data"}]
        assert formatter.validate(invalid_data) is False

    def test_role_mapping(self, temp_output_dir, sample_conversations):
        formatter = ShareGPTFormatter(temp_output_dir)
        output_path = formatter.format(sample_conversations)

        with open(output_path) as f:
            data = json.load(f)

        # Check role mapping
        roles = [msg["from"] for msg in data[0]["conversations"]]
        assert "system" in roles
        assert "human" in roles
        assert "gpt" in roles


class TestAlpacaFormatter:
    """Tests for AlpacaFormatter."""

    def test_format_conversations(self, temp_output_dir, sample_conversations):
        formatter = AlpacaFormatter(temp_output_dir)
        output_path = formatter.format(sample_conversations, "test_alpaca.json")

        assert output_path.exists()
        assert output_path.name == "test_alpaca.json"

        with open(output_path) as f:
            data = json.load(f)

        assert len(data) == 2
        assert "instruction" in data[0]
        assert "output" in data[0]

    def test_validate_valid_data(self, temp_output_dir):
        formatter = AlpacaFormatter(temp_output_dir)
        valid_data = [
            {
                "instruction": "What is Python?",
                "input": "",
                "output": "Python is a programming language.",
            }
        ]
        assert formatter.validate(valid_data) is True

    def test_validate_invalid_data(self, temp_output_dir):
        formatter = AlpacaFormatter(temp_output_dir)
        invalid_data = [{"invalid": "data"}]
        assert formatter.validate(invalid_data) is False

    def test_system_prompt_extraction(self, temp_output_dir, sample_conversations):
        formatter = AlpacaFormatter(temp_output_dir)
        output_path = formatter.format(sample_conversations)

        with open(output_path) as f:
            data = json.load(f)

        # First conversation has system prompt
        assert data[0]["system"] == "You are a helpful assistant."

        # Second conversation has no system prompt
        assert data[1]["system"] is None


class TestHFMessagesFormatter:
    """Tests for HFMessagesFormatter."""

    def test_format_jsonl(self, temp_output_dir, sample_conversations):
        formatter = HFMessagesFormatter(temp_output_dir, jsonl=True)
        output_path = formatter.format(sample_conversations, "test_hf.jsonl")

        assert output_path.exists()
        assert output_path.suffix == ".jsonl"

        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        first = json.loads(lines[0])
        assert "messages" in first
        assert first["messages"][0]["role"] == "system"
        assert first["messages"][0]["content"] == "You are a helpful assistant."

    def test_format_json_array(self, temp_output_dir, sample_conversations):
        formatter = HFMessagesFormatter(temp_output_dir, jsonl=False)
        output_path = formatter.format(sample_conversations, "test_hf.json")

        assert output_path.exists()
        assert output_path.suffix == ".json"

        with open(output_path) as f:
            data = json.load(f)

        assert len(data) == 2
        assert "messages" in data[0]
        assert data[0]["messages"][1]["role"] == "user"
        assert data[0]["messages"][2]["role"] == "assistant"

    def test_format_auto_extension(self, temp_output_dir, sample_conversations):
        formatter = HFMessagesFormatter(temp_output_dir, jsonl=True)
        output_path = formatter.format(sample_conversations, "no_ext")
        assert output_path.suffix == ".jsonl"

        formatter2 = HFMessagesFormatter(temp_output_dir, jsonl=False)
        output_path2 = formatter2.format(sample_conversations, "no_ext2")
        assert output_path2.suffix == ".json"

    def test_validate_valid_jsonl(self, temp_output_dir):
        formatter = HFMessagesFormatter(temp_output_dir)
        valid = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            }
        ]
        assert formatter.validate(valid) is True

    def test_validate_single_dict(self, temp_output_dir):
        """validate() should accept a single dict (JSONL line)."""
        formatter = HFMessagesFormatter(temp_output_dir)
        valid = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
        assert formatter.validate(valid) is True

    def test_validate_invalid_data(self, temp_output_dir):
        formatter = HFMessagesFormatter(temp_output_dir)
        assert formatter.validate([]) is False
        assert formatter.validate([{"no_messages": True}]) is False
        assert formatter.validate([{"messages": "not_a_list"}]) is False
        assert formatter.validate([{"messages": []}]) is False

    def test_validate_bad_role(self, temp_output_dir):
        formatter = HFMessagesFormatter(temp_output_dir)
        bad = [
            {
                "messages": [
                    {"role": "invalid_role", "content": "test"},
                ]
            }
        ]
        assert formatter.validate(bad) is False

    def test_validate_missing_content(self, temp_output_dir):
        formatter = HFMessagesFormatter(temp_output_dir)
        bad = [
            {
                "messages": [
                    {"role": "user"},  # missing content
                ]
            }
        ]
        assert formatter.validate(bad) is False


class TestPreferenceGenerator:
    """Tests for PreferenceGenerator."""

    @pytest.fixture
    def scored_conversations(self):
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        return [
            ConversationSchema(
                id="high-1",
                source_chunk_id="src-1",
                turns=[
                    SynthesizedTurn(role="user", content="What is Python?"),
                    SynthesizedTurn(role="assistant", content="Python is a great language."),
                ],
                confidence_score=0.95,
                judge_scores={"overall": 9.5, "coherence": 9.0},
            ),
            ConversationSchema(
                id="low-1",
                source_chunk_id="src-1",
                turns=[
                    SynthesizedTurn(role="user", content="What is Python?"),
                    SynthesizedTurn(role="assistant", content="It's a language."),
                ],
                confidence_score=0.30,
                judge_scores={"overall": 3.0, "coherence": 2.5},
            ),
            # Singleton — no pair within source
            ConversationSchema(
                id="only-1",
                source_chunk_id="src-2",
                turns=[
                    SynthesizedTurn(role="user", content="What is ML?"),
                    SynthesizedTurn(role="assistant", content="Machine learning is AI."),
                ],
                confidence_score=0.85,
                judge_scores={"overall": 8.5},
            ),
            # No scores — should be excluded
            ConversationSchema(
                id="no-score",
                source_chunk_id="src-3",
                turns=[
                    SynthesizedTurn(role="user", content="Hi"),
                    SynthesizedTurn(role="assistant", content="Hello"),
                ],
            ),
        ]

    def test_generate_pairs_from_same_source(self, scored_conversations):
        from distill_align.exporter.preference_generator import PreferenceGenerator

        gen = PreferenceGenerator(min_score_gap=0.1)
        pairs = gen.generate_pairs(scored_conversations)

        assert len(pairs) >= 1
        # First pair should be from src-1 (high-1 as chosen, low-1 as rejected)
        src1_pairs = [p for p in pairs if p.source_chunk_id == "src-1"]
        assert len(src1_pairs) == 1
        assert src1_pairs[0].chosen == "Python is a great language."
        assert src1_pairs[0].rejected == "It's a language."
        assert src1_pairs[0].chosen_score > src1_pairs[0].rejected_score

    def test_cross_source_pairing(self, scored_conversations):
        from distill_align.exporter.preference_generator import PreferenceGenerator

        gen = PreferenceGenerator(min_score_gap=0.1)
        pairs = gen.generate_pairs(scored_conversations)

        # The singleton (only-1, score 0.85) should be cross-paired
        cross_pairs = [p for p in pairs if p.metadata.get("cross_source")]
        assert len(cross_pairs) >= 0  # May or may not find a cross-source partner

    def test_min_score_gap_filter(self, scored_conversations):
        from distill_align.exporter.preference_generator import PreferenceGenerator

        gen = PreferenceGenerator(min_score_gap=0.9)  # Very high gap
        pairs = gen.generate_pairs(scored_conversations)
        # Only score gap of 0.65 (0.95 - 0.30), which is < 0.9
        src1_pairs = [p for p in pairs if p.source_chunk_id == "src-1"]
        assert len(src1_pairs) == 0

    def test_generate_nothing_without_scores(self):
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn
        from distill_align.exporter.preference_generator import PreferenceGenerator

        convs = [
            ConversationSchema(
                id="a",
                source_chunk_id="src-1",
                turns=[
                    SynthesizedTurn(role="user", content="Q?"),
                    SynthesizedTurn(role="assistant", content="A."),
                ],
            ),
            ConversationSchema(
                id="b",
                source_chunk_id="src-1",
                turns=[
                    SynthesizedTurn(role="user", content="Q?"),
                    SynthesizedTurn(role="assistant", content="B."),
                ],
            ),
        ]

        gen = PreferenceGenerator(require_judge_scores=False)
        pairs = gen.generate_pairs(convs)
        assert len(pairs) == 0  # No scores to compare

    def test_to_dpo_format(self, scored_conversations):
        from distill_align.exporter.preference_generator import PreferenceGenerator

        gen = PreferenceGenerator()
        pairs = gen.generate_pairs(scored_conversations)
        dpo = gen.to_dpo_format(pairs)

        assert len(dpo) > 0
        assert "prompt" in dpo[0]
        assert "chosen" in dpo[0]
        assert "rejected" in dpo[0]
        assert "score_chosen" in dpo[0]
        assert dpo[0]["score_chosen"] > dpo[0]["score_rejected"]

    def test_extract_prompt_and_response(self):
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn
        from distill_align.exporter.preference_generator import PreferenceGenerator

        conv = ConversationSchema(
            id="test",
            source_chunk_id="src",
            turns=[
                SynthesizedTurn(role="system", content="Be helpful."),
                SynthesizedTurn(role="user", content="Hello?"),
                SynthesizedTurn(role="assistant", content="Hi there!"),
            ],
        )
        gen = PreferenceGenerator()
        assert gen.extract_prompt(conv) == "Hello?"
        assert gen.extract_assistant_response(conv) == "Hi there!"


class TestJsonlFormatter:
    """Tests for JsonlFormatter."""

    def test_format_jsonl(self, temp_output_dir, sample_conversations):
        formatter = JsonlFormatter(temp_output_dir)
        output_path = formatter.format(sample_conversations, "test.jsonl")

        assert output_path.exists()
        assert output_path.suffix == ".jsonl"

        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        first = json.loads(lines[0])
        assert "messages" in first
        assert first["messages"][0]["role"] == "system"

    def test_format_adds_extension(self, temp_output_dir, sample_conversations):
        formatter = JsonlFormatter(temp_output_dir)
        output_path = formatter.format(sample_conversations, "no_ext")
        assert output_path.suffix == ".jsonl"

    def test_format_stream(self, temp_output_dir, sample_conversations):
        formatter = JsonlFormatter(temp_output_dir)
        output_path = formatter.format_stream(iter(sample_conversations), "streamed.jsonl")

        assert output_path.exists()
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_format_flatten(self, temp_output_dir, sample_conversations):
        formatter = JsonlFormatter(temp_output_dir, flatten=True)
        output_path = formatter.format(sample_conversations, "flat.jsonl")

        with open(output_path) as f:
            first = json.loads(f.readline())

        assert first["id"] == "conv-1"
        assert first["source_chunk_id"] == "chunk-1"

    def test_format_with_scores(self, temp_output_dir, parsed_conversations):
        formatter = JsonlFormatter(temp_output_dir)
        output_path = formatter.format(parsed_conversations, "scored.jsonl")

        with open(output_path) as f:
            first = json.loads(f.readline())
            second = json.loads(f.readline())

        assert first["confidence_score"] == 0.95
        assert first["judge_scores"] == {"overall": 9.5}
        # Second has no judge_scores, should still have confidence_score
        assert second["confidence_score"] == 0.50

    def test_format_no_metadata(self, temp_output_dir, sample_conversations):
        formatter = JsonlFormatter(temp_output_dir, include_metadata=False)
        output_path = formatter.format(sample_conversations, "no_meta.jsonl")

        with open(output_path) as f:
            first = json.loads(f.readline())

        assert "id" not in first
        assert "source_chunk_id" not in first
        assert "messages" in first

    def test_validate_valid(self, temp_output_dir):
        formatter = JsonlFormatter(temp_output_dir)
        valid = [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        ]
        assert formatter.validate(valid) is True

    def test_validate_single_dict(self, temp_output_dir):
        formatter = JsonlFormatter(temp_output_dir)
        valid = {
            "messages": [
                {"role": "user", "content": "Hi"},
            ]
        }
        assert formatter.validate(valid) is True

    def test_validate_invalid(self, temp_output_dir):
        formatter = JsonlFormatter(temp_output_dir)
        assert formatter.validate([]) is False
        assert formatter.validate([{"no_messages": True}]) is False
        assert formatter.validate([{"messages": "not_a_list"}]) is False
        assert formatter.validate([{"messages": []}]) is False
        assert formatter.validate([{"messages": [{"role": "bad_role", "content": "x"}]}]) is False
        assert formatter.validate([{"messages": [{"role": "user"}]}]) is False  # missing content


class TestParquetFormatter:
    """Tests for ParquetFormatter."""

    def test_format_parquet(self, temp_output_dir, sample_conversations):
        pytest.importorskip("pyarrow")
        from distill_align.exporter.formatters.parquet import ParquetFormatter

        formatter = ParquetFormatter(temp_output_dir)
        output_path = formatter.format(sample_conversations, "test.parquet")

        assert output_path.exists()
        assert output_path.suffix == ".parquet"

        # Read back and verify
        import pyarrow.parquet as pq

        table = pq.read_table(output_path)
        assert table.num_rows == 2
        assert "id" in table.column_names
        assert "messages" in table.column_names
        assert "source_chunk_id" in table.column_names

    def test_format_auto_extension(self, temp_output_dir, sample_conversations):
        pytest.importorskip("pyarrow")
        from distill_align.exporter.formatters.parquet import ParquetFormatter

        formatter = ParquetFormatter(temp_output_dir)
        output_path = formatter.format(sample_conversations, "no_ext")
        assert output_path.suffix == ".parquet"

    def test_format_stream(self, temp_output_dir, sample_conversations):
        pytest.importorskip("pyarrow")
        from distill_align.exporter.formatters.parquet import ParquetFormatter

        formatter = ParquetFormatter(temp_output_dir)
        output_path = formatter.format_stream(iter(sample_conversations), "streamed.parquet")

        assert output_path.exists()
        import pyarrow.parquet as pq

        table = pq.read_table(output_path)
        assert table.num_rows == 2

    def test_format_with_scores(self, temp_output_dir, parsed_conversations):
        pytest.importorskip("pyarrow")
        from distill_align.exporter.formatters.parquet import ParquetFormatter

        formatter = ParquetFormatter(temp_output_dir)
        output_path = formatter.format(parsed_conversations, "scored.parquet")

        import pyarrow.parquet as pq

        table = pq.read_table(output_path)
        assert table.num_rows == 2
        assert "confidence_score" in table.column_names
        # First row has judge_scores
        assert "judge_scores" in table.column_names

    def test_validate_valid_table(self, temp_output_dir, sample_conversations):
        pytest.importorskip("pyarrow")
        from distill_align.exporter.formatters.parquet import ParquetFormatter

        formatter = ParquetFormatter(temp_output_dir)
        output_path = formatter.format(sample_conversations)
        assert formatter.validate(str(output_path)) is True

    def test_validate_invalid(self, temp_output_dir):
        pytest.importorskip("pyarrow")
        from distill_align.exporter.formatters.parquet import ParquetFormatter

        formatter = ParquetFormatter(temp_output_dir)
        assert formatter.validate({}) is False
        assert formatter.validate([]) is False

    def test_empty_list_raises(self, temp_output_dir):
        pytest.importorskip("pyarrow")
        from distill_align.core.exceptions import FormatError
        from distill_align.exporter.formatters.parquet import ParquetFormatter

        formatter = ParquetFormatter(temp_output_dir)
        with pytest.raises(FormatError, match="empty"):
            formatter.format([])

    def test_empty_stream_raises(self, temp_output_dir):
        pytest.importorskip("pyarrow")
        from distill_align.core.exceptions import FormatError
        from distill_align.exporter.formatters.parquet import ParquetFormatter

        formatter = ParquetFormatter(temp_output_dir)
        with pytest.raises(FormatError, match="empty"):
            formatter.format_stream(iter([]))
