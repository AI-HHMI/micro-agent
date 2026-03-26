"""Base class for dataset scanners."""

from __future__ import annotations

from abc import ABC, abstractmethod

from micro_agent.discover import DiscoveredDataset


class BaseScanner(ABC):
    """Abstract base for repository-specific dataset scanners.

    Each scanner knows how to query a specific data source and return
    DiscoveredDataset entries with populated metadata.
    """

    name: str = "base"

    @abstractmethod
    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        """Scan the source and return discovered datasets."""
        ...

    @abstractmethod
    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        """Test that a discovered dataset is actually accessible.

        Returns:
            "verified" if the data can be read anonymously,
            "failed" if not accessible,
            "pending" if validation could not be completed.
        """
        ...
