"""Configuration types for mlxmolkit."""

from dataclasses import dataclass


@dataclass
class EmbedConfig:
    """Configuration for conformer embedding."""

    batch_size: int = 64
    num_workers: int = 1
