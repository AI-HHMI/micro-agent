"""Dataset accessibility validation.

Verifies that discovered datasets can actually be read anonymously
before adding them to the registry catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from trailhead.discover import DiscoveredDataset


@dataclass
class ValidationResult:
    """Result of validating a discovered dataset."""

    status: str  # "verified" | "failed" | "pending"
    accessible: bool = False
    metadata: dict = field(default_factory=dict)
    error: str = ""


async def validate_url_reachable(url: str, timeout: float = 15.0) -> bool:
    """Check if a URL is reachable with a HEAD request."""
    if not url or url.startswith("s3://") or url.startswith("gs://"):
        return True  # S3/GCS paths need backend-specific validation
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.head(url, follow_redirects=True)
            return resp.status_code < 400
    except Exception:
        return False


async def validate_dataset(dataset: DiscoveredDataset) -> ValidationResult:
    """Validate that a discovered dataset is accessible.

    Performs:
    1. URL reachability check (HEAD request)
    2. Metadata sanity checks (has title, has format)
    3. Sets validation_status on the dataset

    Returns a ValidationResult with details.
    """
    errors: list[str] = []

    # Check URL is reachable
    url = dataset.access_url
    if url and not url.startswith(("s3://", "gs://")):
        reachable = await validate_url_reachable(url)
        if not reachable:
            errors.append(f"URL not reachable: {url}")

    # Metadata sanity checks
    if not dataset.title or dataset.title == dataset.id:
        errors.append("Missing or generic title")

    if not dataset.data_format:
        errors.append("Missing data format")

    if errors:
        result = ValidationResult(
            status="failed",
            accessible=len(errors) == 1 and "title" in errors[0],
            error="; ".join(errors),
        )
    else:
        result = ValidationResult(
            status="verified",
            accessible=True,
            metadata={
                "id": dataset.id,
                "repository": dataset.repository,
                "data_format": dataset.data_format,
                "modality_class": dataset.modality_class,
            },
        )

    dataset.validation_status = result.status
    return result


async def validate_batch(
    datasets: list[DiscoveredDataset],
) -> list[ValidationResult]:
    """Validate a batch of discovered datasets."""
    import asyncio
    tasks = [validate_dataset(ds) for ds in datasets]
    return await asyncio.gather(*tasks)
