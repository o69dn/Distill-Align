"""
Configuration file support for Distill-Align.

Loads project settings from distill-align.yaml or distill-align.toml.
"""

from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Project-level configuration."""

    name: str = "my-dataset"
    version: str = "1.0"
    description: str = ""


class SourceConfig(BaseModel):
    """A single ingestion source."""

    path: str
    type: str = "auto"  # auto, markdown, code, pdf, docx, html, jupyter, json, csv, text
    recursive: bool = True
    patterns: list[str] = Field(default_factory=list)


class IngestionFileConfig(BaseModel):
    """Ingestion configuration from file."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    respect_headers: bool = True
    max_chunk_tokens: int = 4000
    sources: list[SourceConfig] = Field(default_factory=list)


class SynthesisFileConfig(BaseModel):
    """Synthesis configuration from file."""

    provider: str = "openai"
    model: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = None
    max_concurrency: int = 5
    max_rpm: int = 60
    temperature: float = 0.7
    socratic: bool = True
    scaffold: bool = True
    retry_attempts: int = 5


class UnslothFileConfig(BaseModel):
    """Unsloth configuration from file."""

    model: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    max_seq_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 16
    load_in_4bit: bool = True
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4


class ExportFileConfig(BaseModel):
    """Export configuration from file."""

    formats: list[str] = Field(default_factory=lambda: ["sharegpt"])
    output_dir: str = "./output"
    generate_unsloth_script: bool = True
    unsloth: UnslothFileConfig = Field(default_factory=UnslothFileConfig)
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05


class DistillAlignConfig(BaseModel):
    """Full Distill-Align configuration from file."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    ingestion: IngestionFileConfig = Field(default_factory=IngestionFileConfig)
    synthesis: SynthesisFileConfig = Field(default_factory=SynthesisFileConfig)
    export: ExportFileConfig = Field(default_factory=ExportFileConfig)
    log_level: str = "INFO"
    cache_dir: str = ".cache"


# Default config template for init wizard
DEFAULT_CONFIG_TEMPLATE = """# Distill-Align Configuration
# See https://github.com/o69dn/Distill-Align for documentation

project:
  name: "{project_name}"
  version: "1.0"
  description: ""

ingestion:
  chunk_size: 1000
  chunk_overlap: 200
  respect_headers: true
  max_chunk_tokens: 4000
  sources:
    - path: ./data
      type: auto
      recursive: true

synthesis:
  provider: openai        # openai, ollama, vllm
  model: gpt-4o
  # base_url: http://localhost:11434  # For Ollama/vLLM
  # api_key: sk-...                    # Or set OPENAI_API_KEY env var
  max_concurrency: 5
  max_rpm: 60
  temperature: 0.7
  socratic: true
  scaffold: true
  retry_attempts: 5

export:
  formats:
    - sharegpt
    - alpaca
  output_dir: ./output
  generate_unsloth_script: true
  train_split: 0.9
  val_split: 0.05
  test_split: 0.05
  unsloth:
    model: unsloth/Meta-Llama-3.1-8B-Instruct
    max_seq_length: 2048
    lora_rank: 16
    lora_alpha: 16
    load_in_4bit: true
    batch_size: 2
    gradient_accumulation_steps: 4
    num_epochs: 3
    learning_rate: 0.0002

log_level: INFO
cache_dir: .cache
"""


def find_config_file(start_dir: str | Path = ".") -> Path | None:
    """
    Search for a config file starting from the given directory.

    Looks for distill-align.yaml, distill-align.yml, or distill-align.toml.

    Args:
        start_dir: Directory to start searching from.

    Returns:
        Path to config file or None.
    """
    start = Path(start_dir).resolve()

    for name in ["distill-align.yaml", "distill-align.yml", "distill-align.toml"]:
        path = start / name
        if path.exists():
            return path

    # Also check parent directories (up to 3 levels)
    current = start
    for _ in range(3):
        current = current.parent
        if current == current.parent:
            break
        for name in ["distill-align.yaml", "distill-align.yml", "distill-align.toml"]:
            path = current / name
            if path.exists():
                return path

    return None


def load_config(config_path: str | Path | None = None) -> DistillAlignConfig:
    """
    Load configuration from file.

    Args:
        config_path: Explicit path to config file. Auto-detects if None.

    Returns:
        DistillAlignConfig instance.
    """
    path: Path | None
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
    else:
        path = find_config_file()

    if path is None:
        logger.debug("No config file found, using defaults")
        return DistillAlignConfig()

    logger.info(f"Loading config from {path}")

    if path.suffix == ".toml":
        with open(path, "rb") as f:
            try:
                import tomllib

                data = tomllib.load(f)
            except ImportError:
                import tomli  # type: ignore[import-not-found]

                data = tomli.load(f)
    else:
        with open(path, encoding="utf-8") as f:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML config files. Install with: pip install pyyaml"
                ) from None
            data = yaml.safe_load(f)

    if data is None:
        return DistillAlignConfig()

    return DistillAlignConfig(**data)


def save_config(config: DistillAlignConfig, path: str | Path = "distill-align.yaml") -> Path:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration to save.
        path: Output file path.

    Returns:
        Path to the saved file.
    """
    output_path = Path(path)
    data = config.model_dump()

    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for saving YAML config files. Install with: pip install pyyaml") from None

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"Saved config to {output_path}")
    return output_path


def generate_default_config(project_name: str = "my-dataset", path: str = "distill-align.yaml") -> Path:
    """
    Generate a default configuration file.

    Args:
        project_name: Name for the project.
        path: Output file path.

    Returns:
        Path to the generated file.
    """
    output_path = Path(path)
    content = DEFAULT_CONFIG_TEMPLATE.replace("{project_name}", project_name)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Generated default config at {output_path}")
    return output_path
