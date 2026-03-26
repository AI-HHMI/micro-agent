"""Tool definitions and executors for the discovery agent.

These are the tools the LLM can call during a discovery cycle.
Each tool wraps a scanner, validator, or utility function.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import httpx

from trailhead.agent.llm import ToolDefinition
from trailhead.discover import DiscoveredDataset
from trailhead.registry import Registry
from trailhead.validate import validate_dataset


# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema for LLM)
# ---------------------------------------------------------------------------

SCAN_REPOSITORY = ToolDefinition(
    name="scan_repository",
    description=(
        "Scan a specific data repository for microscopy datasets. "
        "Available repositories: OpenOrganelle, EMPIAR, IDR, BioImage Archive, "
        "Allen, HPA, CellImageLibrary, Zenodo."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repository": {
                "type": "string",
                "description": "Name of the repository to scan",
                "enum": [
                    "OpenOrganelle", "EMPIAR", "IDR", "BioImage Archive",
                    "Allen", "HPA", "CellImageLibrary", "Zenodo",
                ],
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of datasets to return",
                "default": 25,
            },
        },
        "required": ["repository"],
    },
)

WEB_SEARCH = ToolDefinition(
    name="web_search",
    description=(
        "Search the web for recently published microscopy datasets, papers, "
        "or data announcements. Returns text snippets from search results."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g., 'new public light sheet dataset 2024')",
            },
        },
        "required": ["query"],
    },
)

FETCH_WEBPAGE = ToolDefinition(
    name="fetch_webpage",
    description=(
        "Fetch and extract text content from a URL. Useful for reading "
        "dataset landing pages, paper abstracts, or repository listings."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
        },
        "required": ["url"],
    },
)

VALIDATE_DATASET = ToolDefinition(
    name="validate_dataset",
    description=(
        "Validate that a discovered dataset is accessible. "
        "Checks URL reachability and metadata completeness."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dataset": {
                "type": "object",
                "description": "The dataset to validate (DiscoveredDataset fields)",
            },
        },
        "required": ["dataset"],
    },
)

CHECK_EXISTING = ToolDefinition(
    name="check_existing",
    description=(
        "Check if a dataset already exists in the registry. "
        "Returns the existing entry if found, or null if not."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dataset_id": {
                "type": "string",
                "description": "The dataset ID to check",
            },
        },
        "required": ["dataset_id"],
    },
)

SAVE_CANDIDATE = ToolDefinition(
    name="save_candidate",
    description=(
        "Save a discovered dataset candidate to the results list. "
        "The dataset will be saved with its current validation_status."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dataset": {
                "type": "object",
                "description": "The dataset to save (DiscoveredDataset fields)",
            },
        },
        "required": ["dataset"],
    },
)

ALL_TOOLS = [
    SCAN_REPOSITORY,
    WEB_SEARCH,
    FETCH_WEBPAGE,
    VALIDATE_DATASET,
    CHECK_EXISTING,
    SAVE_CANDIDATE,
]


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Executes tools called by the LLM agent."""

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.candidates: list[DiscoveredDataset] = []
        self._scanner_cache: dict = {}

    async def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result as a string."""
        handler = {
            "scan_repository": self._scan_repository,
            "web_search": self._web_search,
            "fetch_webpage": self._fetch_webpage,
            "validate_dataset": self._validate_dataset,
            "check_existing": self._check_existing,
            "save_candidate": self._save_candidate,
        }.get(tool_name)

        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = await handler(arguments)
            return result
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _scan_repository(self, args: dict) -> str:
        from trailhead.scanners import (
            OpenOrganelleScanner, EMPIARScanner, IDRScanner,
            BioImageArchiveScanner, AllenScanner, HPAScanner,
            CellImageLibraryScanner, ZenodoScanner,
        )

        scanner_map = {
            "OpenOrganelle": OpenOrganelleScanner,
            "EMPIAR": EMPIARScanner,
            "IDR": IDRScanner,
            "BioImage Archive": BioImageArchiveScanner,
            "Allen": AllenScanner,
            "HPA": HPAScanner,
            "CellImageLibrary": CellImageLibraryScanner,
            "Zenodo": ZenodoScanner,
        }

        repo = args["repository"]
        limit = args.get("limit", 25)
        scanner_cls = scanner_map.get(repo)
        if not scanner_cls:
            return json.dumps({"error": f"Unknown repository: {repo}"})

        scanner = scanner_cls()
        datasets = await scanner.scan(limit=limit)
        return json.dumps({
            "repository": repo,
            "count": len(datasets),
            "datasets": [asdict(d) for d in datasets[:10]],  # Truncate for LLM context
            "total_available": len(datasets),
        })

    async def _web_search(self, args: dict) -> str:
        query = args["query"]
        # Use a simple web search via httpx — DuckDuckGo lite
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://lite.duckduckgo.com/lite/",
                    params={"q": query},
                    follow_redirects=True,
                )
                if resp.status_code == 200:
                    # Extract text snippets from HTML (simple parsing)
                    text = resp.text
                    # Remove HTML tags for a rough text extraction
                    import re
                    clean = re.sub(r"<[^>]+>", " ", text)
                    clean = re.sub(r"\s+", " ", clean).strip()
                    return json.dumps({
                        "query": query,
                        "results": clean[:3000],  # Truncate for LLM context
                    })
        except Exception as e:
            return json.dumps({"query": query, "error": str(e)})
        return json.dumps({"query": query, "results": "No results found"})

    async def _fetch_webpage(self, args: dict) -> str:
        url = args["url"]
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, follow_redirects=True)
                if resp.status_code == 200:
                    import re
                    clean = re.sub(r"<[^>]+>", " ", resp.text)
                    clean = re.sub(r"\s+", " ", clean).strip()
                    return json.dumps({
                        "url": url,
                        "content": clean[:3000],
                    })
                return json.dumps({"url": url, "error": f"HTTP {resp.status_code}"})
        except Exception as e:
            return json.dumps({"url": url, "error": str(e)})

    async def _validate_dataset(self, args: dict) -> str:
        ds_dict = args["dataset"]
        ds = DiscoveredDataset(**{
            k: v for k, v in ds_dict.items()
            if k in DiscoveredDataset.__dataclass_fields__
        })
        result = await validate_dataset(ds)
        return json.dumps({
            "status": result.status,
            "accessible": result.accessible,
            "error": result.error,
        })

    async def _check_existing(self, args: dict) -> str:
        dataset_id = args["dataset_id"]
        matches = [e for e in self.registry.entries if e.id == dataset_id]
        if matches:
            entry = matches[0]
            return json.dumps({
                "exists": True,
                "id": entry.id,
                "repository": entry.repository,
                "title": entry.title,
            })
        return json.dumps({"exists": False})

    async def _save_candidate(self, args: dict) -> str:
        ds_dict = args["dataset"]
        ds = DiscoveredDataset(**{
            k: v for k, v in ds_dict.items()
            if k in DiscoveredDataset.__dataclass_fields__
        })
        # Deduplicate
        existing_ids = {c.id for c in self.candidates}
        if ds.id in existing_ids:
            return json.dumps({"saved": False, "reason": "Already in candidates"})
        self.candidates.append(ds)
        return json.dumps({"saved": True, "id": ds.id, "total_candidates": len(self.candidates)})
