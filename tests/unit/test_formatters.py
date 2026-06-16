"""
Unit tests for exporter module.
"""

import json

import pytest
from distill_align.core.schemas import ConversationSchema, SynthesizedTurn
from distill_align.exporter.formatters.alpaca import AlpacaFormatter
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
