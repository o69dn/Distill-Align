"""
HuggingFace Hub integration for dataset publishing.

Handles authentication, dataset creation, file upload, and card generation.
"""

from pathlib import Path

from loguru import logger

from ..core.exceptions import ExportError


class HFDatasetPublisher:
    """Publishes datasets to HuggingFace Hub."""

    def __init__(self, token: str | None = None, repo_id: str | None = None):
        """Initialize the HF Hub publisher.

        Args:
            token: HuggingFace API token. Uses HF_TOKEN env var if not provided.
            repo_id: Default repository ID (e.g. "username/dataset-name").
        """
        self.token = token
        self.repo_id = repo_id

    def push(
        self,
        files: dict[str, Path],
        repo_id: str | None = None,
        message: str = "Update dataset",
        private: bool = False,
    ) -> bool:
        """Push dataset files to HuggingFace Hub.

        Args:
            files: Dictionary mapping file types to file paths.
            repo_id: Target repository ID (overrides instance default).
            message: Commit message for the upload.
            private: Whether the dataset should be private.

        Returns:
            True if successful.

        Raises:
            ExportError: If the push fails.
        """
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return False

        repo_id = repo_id or self.repo_id
        if not repo_id:
            raise ExportError("repo_id is required to push to HuggingFace Hub")

        try:
            api = HfApi(token=self.token)

            # Create repo if it doesn't exist
            try:
                api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create repo: {e}")

            # Upload files
            for file_type, file_path in files.items():
                if file_path.exists():
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=repo_id,
                        commit_message=f"{message}: {file_type}",
                    )
                    logger.info(f"Uploaded {file_type} to {repo_id}")

            return True
        except Exception as e:
            raise ExportError(f"Failed to push to HuggingFace Hub: {e}") from e

    @staticmethod
    def check_installed() -> bool:
        """Check if huggingface_hub is installed.

        Returns:
            True if huggingface_hub is available.
        """
        try:
            import huggingface_hub  # noqa: F401

            return True
        except ImportError:
            return False
