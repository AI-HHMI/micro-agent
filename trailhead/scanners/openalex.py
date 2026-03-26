"""Scanner for OpenAlex — academic literature-based dataset discovery.

OpenAlex indexes >250M scholarly works. This scanner searches for papers
that mention microscopy datasets and extracts DOI/URL links that may point
to deposited data (Zenodo, Figshare, Dryad, institutional repos).
No API key required — polite pool allows ~10 req/s with mailto.

API docs: https://docs.openalex.org/
"""

from __future__ import annotations

import re

import httpx

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner

OPENALEX_API = "https://api.openalex.org/works"

# Queries that tend to surface papers with deposited imaging data
QUERIES = [
    ("fluorescence microscopy open dataset", "fluorescence"),
    ("electron microscopy segmentation public dataset", "em"),
    ("cell painting public dataset", "fluorescence"),
    ("light sheet microscopy dataset", "fluorescence"),
    ("volume electron microscopy connectome", "em"),
    ("confocal microscopy image dataset", "fluorescence"),
]

# Domains likely to host actual data
DATA_DOMAINS = {
    "zenodo.org", "figshare.com", "datadryad.org", "dryad.org",
    "dataverse.harvard.edu", "osf.io", "huggingface.co",
    "idr.openmicroscopy.org", "empiar.org", "ebi.ac.uk",
    "quiltdata.com", "s3.amazonaws.com", "github.com",
    "cellimages.library", "proteinatlas.org",
}


def _extract_data_urls(abstract: str) -> list[str]:
    """Pull URLs from abstract text that likely point to datasets."""
    urls = re.findall(r'https?://[^\s<>"\')\]]+', abstract)
    data_urls = []
    for url in urls:
        url = url.rstrip(".,;:")
        if any(domain in url for domain in DATA_DOMAINS):
            data_urls.append(url)
    return data_urls


class OpenAlexScanner(BaseScanner):
    """Discover microscopy datasets by mining academic literature via OpenAlex."""

    name = "OpenAlex"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        results: list[DiscoveredDataset] = []
        seen_dois: set[str] = set()

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for query, modality_class in QUERIES:
                if len(results) >= limit:
                    break
                try:
                    resp = await client.get(
                        OPENALEX_API,
                        params={
                            "search": query,
                            "filter": "is_oa:true,from_publication_date:2020-01-01",
                            "sort": "cited_by_count:desc",
                            "per_page": min(limit, 25),
                            "mailto": "trailhead-scanner@example.com",
                        },
                    )
                    if resp.status_code != 200:
                        continue

                    works = resp.json().get("results", [])

                    for work in works:
                        doi = work.get("doi", "") or ""
                        if doi in seen_dois:
                            continue
                        if doi:
                            seen_dois.add(doi)

                        title = work.get("title", "") or ""
                        abstract_inv = work.get("abstract_inverted_index") or {}

                        # Reconstruct abstract from inverted index
                        if abstract_inv:
                            word_positions: list[tuple[int, str]] = []
                            for word, positions in abstract_inv.items():
                                for pos in positions:
                                    word_positions.append((pos, word))
                            word_positions.sort()
                            abstract = " ".join(w for _, w in word_positions)
                        else:
                            abstract = ""

                        # Look for data URLs in abstract
                        data_urls = _extract_data_urls(abstract)

                        # Extract organism from concepts/topics
                        organism = ""
                        concepts = work.get("concepts", []) or []
                        for c in concepts:
                            name = (c.get("display_name", "") or "").lower()
                            if any(org in name for org in [
                                "homo sapiens", "human", "mouse", "mus musculus",
                                "drosophila", "c. elegans", "zebrafish",
                            ]):
                                organism = c.get("display_name", "")
                                break

                        # Use first data URL as access point, or fall back to DOI
                        access_url = data_urls[0] if data_urls else doi

                        if not access_url:
                            continue

                        work_id = work.get("id", "").split("/")[-1] if work.get("id") else ""

                        results.append(DiscoveredDataset(
                            id=f"openalex_{work_id}" if work_id else f"openalex_{len(results)}",
                            repository="OpenAlex",
                            title=title[:120] if title else "Unknown",
                            organism=organism,
                            imaging_modality=query.split(" dataset")[0].split(" public")[0],
                            has_raw=bool(data_urls),
                            data_format="",
                            access_url=access_url,
                            provenance=f"OpenAlex literature search for '{query}'; DOI: {doi}",
                            modality_class=modality_class,
                            supports_random_access=False,
                            validation_status="pending",
                        ))

                except Exception as e:
                    print(f"  [{self.name}] API error for '{query}': {e}")

        return results[:limit]

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        """Check if the data URL is reachable."""
        if not dataset.access_url or dataset.access_url.startswith("https://doi.org"):
            return "pending"  # DOI links need further resolution
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.head(dataset.access_url, follow_redirects=True)
                return "verified" if resp.status_code < 400 else "pending"
        except Exception:
            return "pending"
