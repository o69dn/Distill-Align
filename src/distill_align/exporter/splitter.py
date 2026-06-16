"""
Dataset splitter for train/val/test splits.

Provides reproducible, optionally stratified dataset splitting.
"""

import random
from pathlib import Path

from loguru import logger

from ..core.schemas import ConversationSchema


class DatasetSplit:
    """Container for split datasets."""

    def __init__(
        self,
        train: list[ConversationSchema],
        val: list[ConversationSchema],
        test: list[ConversationSchema],
    ):
        """
        Initialize the dataset split.

        Args:
            train: Training conversations.
            val: Validation conversations.
            test: Test conversations.
        """
        self.train = train
        self.val = val
        self.test = test

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "total": self.total,
        }


class DatasetSplitter:
    """
    Splits datasets into train/val/test sets.

    Supports:
    - Reproducible splits via random seed
    - Configurable ratios
    - Optional stratification by source type
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the splitter.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed

    def split(
        self,
        conversations: list[ConversationSchema],
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        stratify_by: str | None = None,
    ) -> DatasetSplit:
        """
        Split conversations into train/val/test sets.

        Args:
            conversations: Conversations to split.
            train_ratio: Training set ratio (0-1).
            val_ratio: Validation set ratio (0-1).
            test_ratio: Test set ratio (0-1).
            stratify_by: Optional field to stratify by ('source_chunk_id', etc.).

        Returns:
            DatasetSplit with train/val/test lists.

        Raises:
            ValueError: If ratios don't sum to 1.0.
        """
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total:.3f}")

        if not conversations:
            return DatasetSplit([], [], [])

        rng = random.Random(self.seed)

        if stratify_by:
            return self._stratified_split(conversations, train_ratio, val_ratio, test_ratio, stratify_by, rng)
        else:
            return self._random_split(conversations, train_ratio, val_ratio, test_ratio, rng)

    def _random_split(
        self,
        conversations: list[ConversationSchema],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        rng: random.Random,
    ) -> DatasetSplit:
        """Random split without stratification."""
        indices = list(range(len(conversations)))
        rng.shuffle(indices)

        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        return DatasetSplit(
            train=[conversations[i] for i in train_indices],
            val=[conversations[i] for i in val_indices],
            test=[conversations[i] for i in test_indices],
        )

    def _stratified_split(
        self,
        conversations: list[ConversationSchema],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        stratify_by: str,
        rng: random.Random,
    ) -> DatasetSplit:
        """Stratified split maintaining distribution of stratification field."""
        # Group by stratification field
        groups: dict[str, list[ConversationSchema]] = {}
        for conv in conversations:
            key = conv.source_chunk_id if stratify_by == "source_chunk_id" else getattr(conv, stratify_by, "unknown")

            if key not in groups:
                groups[key] = []
            groups[key].append(conv)

        train_list: list[ConversationSchema] = []
        val_list: list[ConversationSchema] = []
        test_list: list[ConversationSchema] = []

        # Split each group proportionally
        for _key, group in groups.items():
            rng.shuffle(group)
            n = len(group)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_list.extend(group[:train_end])
            val_list.extend(group[train_end:val_end])
            test_list.extend(group[val_end:])

        # Shuffle the final lists to mix groups
        rng.shuffle(train_list)
        rng.shuffle(val_list)
        rng.shuffle(test_list)

        return DatasetSplit(
            train=train_list,
            val=val_list,
            test=test_list,
        )

    def save_split(
        self,
        split: DatasetSplit,
        output_dir: str | Path,
        prefix: str = "dataset",
    ) -> dict[str, Path]:
        """
        Save split datasets to JSON files.

        Args:
            split: DatasetSplit to save.
            output_dir: Output directory.
            prefix: Filename prefix.

        Returns:
            Dictionary mapping split name to file path.
        """
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        paths = {}
        for name, convs in [("train", split.train), ("val", split.val), ("test", split.test)]:
            if not convs:
                continue
            file_path = output_path / f"{prefix}_{name}.json"
            data = [c.model_dump() for c in convs]
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            paths[name] = file_path
            logger.info(f"Saved {name} split: {len(convs)} conversations to {file_path}")

        return paths
