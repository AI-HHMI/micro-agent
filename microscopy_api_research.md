# Microscopy Repository API Research

Prepared for meeting with Anthropic engineers on building agentic discovery over external microscopy repositories.

---

## 1. Repository API Findings

### EMPIAR (Electron Microscopy Public Image Archive)

| Fact | Detail |
|---|---|
| **Base URL** | `https://www.ebi.ac.uk/empiar/api/` |
| **Docs** | `https://www.ebi.ac.uk/empiar/api/documentation/` (Django REST Framework browsable) |
| **Auth** | None for reads. API token for depositions (`/deposition/api_token`) |
| **Protocol** | REST (Django REST Framework). No GraphQL. |
| **Key endpoints** | `GET /entry/EMPIAR-{ID}/` (single entry), `POST /entry/` (batch), `GET /emdb_entry/{EMDB_ID}/` (cross-ref), `GET /empiar_citations/` (latest pubs) |
| **Python clients** | `empiar-depositor` (deposition CLI), `empiarreader` (lazy-load MRC/STAR into xarray/dask) |
| **Data formats** | MRC, MRCS, TIFF, DM4, EER, IMAGIC, SPIDER. API returns JSON. |
| **Rate limits** | No explicit limits. EBI fair-use (~20 req/s). Downloads >4GB require Globus/Aspera. |

**Gaps:**
- **No search/filter endpoint** — cannot search by keyword, organism, author, experiment type via API
- **No list-all/pagination endpoint** — cannot enumerate entries programmatically
- **No file-level metadata** via REST — only entry-level
- **No bulk download URLs** in API responses (must use Aspera/Globus separately)

---

### IDR (Image Data Resource)

| Fact | Detail |
|---|---|
| **Base URL** | `https://idr.openmicroscopy.org` |
| **Docs** | `/about/api.html`, OMERO JSON API docs, Search Engine Swagger at `/searchengine/apidocs/` |
| **Auth** | None for reads (public read-only). BlitzGateway uses `public`/`public`. |
| **Protocol** | REST (3 layers: OMERO JSON API, Webclient API, WebGateway). No GraphQL. |
| **Key endpoints** | `/api/v0/m/projects/`, `/api/v0/m/datasets/{id}/images/`, `/api/v0/m/screens/{id}/plates/`, `/webclient/api/annotations/?type=map&image={id}`, `/webgateway/render_image/{id}/{z}/{t}/`, `/mapr/api/{type}/?value={val}` (gene/phenotype/organism queries) |
| **Python clients** | `idr-py` (wraps both BlitzGateway + HTTP), `omero-py` (core OMERO bindings), `ome-zarr` |
| **Data formats** | OME-TIFF, OME-Zarr on S3 (`uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/`), JPEG renders, JSON metadata |
| **Rate limits** | None published. Read-only. No original file download via web (use FTP/Aspera/Globus). |

**Gaps:**
- **No raw pixel access via REST** — need BlitzGateway Ice protocol
- **No original file download via web API** — need FTP/Aspera/Globus
- **No arbitrary HQL queries** via REST
- **No batch annotation export** in one call
- **MAPR queries limited** to pre-indexed annotation types (gene, phenotype, organism, compound)

---

### MICrONS (Allen Institute Connectomics)

| Fact | Detail |
|---|---|
| **Base URL** | `https://global.daf-apis.com` (CAVE server) |
| **Docs** | `caveclient.readthedocs.io`, `tutorial.microns-explorer.org` |
| **Auth** | Token-based. `client.auth.setup_token(make_new=True)` opens browser flow. Stored at `~/.cloudvolume/secrets/cave-secret.json`. **Required even for public data.** |
| **Protocol** | REST microservices under CAVE. No GraphQL. |
| **Key endpoints** | `/materialize/` (annotation queries), `/segmentation/` (chunkedgraph), `/annotation/`, `/nglstate/api/v1/` (neuroglancer states), `/l2cache/`, `/info/`, `/schema/` |
| **Python clients** | `caveclient` (primary), `cloud-volume` (Neuroglancer precomputed volumes), `pcg-skel` (skeletons), `nglui` (neuroglancer state builder), `meshparty` (mesh/skeleton data) |
| **Data formats** | Neuroglancer Precomputed (imagery/seg/mesh), PostgreSQL/PostGIS (annotations as DataFrames), BigTable (chunkedgraph). On GCS + AWS S3. |
| **Rate limits** | None published. Practical: synapses table has ~337M rows, unfiltered queries time out. |

**Gaps:**
- **No spatial bounding-box queries** on materialization (must use CloudVolume or coordinate filters)
- **No streaming/pub-sub** for proofreading changes
- **No direct SQL access** to materialization database
- **Public users are read-only** — no proofreading operations
- **Functional imaging data** (DataJoint pipelines) not exposed through CAVE
- **Auth required even for public data** — friction for discovery tools

---

### BioImage Archive (BIA)

| Fact | Detail |
|---|---|
| **Base URL** | BioStudies API: `https://www.ebi.ac.uk/biostudies/api/v1/`, BIA Integrator: separate FastAPI (not well-documented public URL) |
| **Docs** | `biostudies.gitbook.io/biostudies-api`, `bia-integrator.readthedocs.io` |
| **Auth** | None for reads. BioStudies submission: session login. BIA Integrator v2: bearer token (`POST /v2/auth/token`). |
| **Protocol** | REST. No GraphQL. |
| **Key endpoints** | `GET /biostudies/api/v1/studies/{accNo}` (full study JSON), `GET .../info` (metadata+FTP link), S3 access: `aws --endpoint-url https://uk1s3.embassy.ebi.ac.uk s3 ls s3://bia-integrator-data/S-BIAD{id}/` |
| **Python clients** | `biostudies-client` (BioStudies wrapper), `bia-explorer` (Jupyter exploration), `bia-integrator` (monorepo: shared data models, API, ingest, search, export) |
| **Data formats** | OME-Zarr (NGFF), OME-TIFF, proprietary. Metadata: REMBI, MIFA (LinkML). Download via FTP/Aspera/S3/Globus. |
| **Rate limits** | None explicit. EBI fair-use. |

**Gaps:**
- **No single unified public REST API** — split across BioStudies (study-level) and BIA Integrator (image-level, primarily internal)
- **No search-by-image-content** (organism, modality, resolution) via REST
- **No bulk metadata export** for all studies
- **Pagination/filtering poorly documented** in BioStudies API
- **BIA Integrator API not reliably public** — internal tooling without stable public endpoint

---

### OpenOrganelle (Janelia/CellMap)

| Fact | Detail |
|---|---|
| **Base URL** | **No REST API exists.** React frontend reads directly from S3. |
| **Docs** | `openorganelle.janelia.org` (web portal only) |
| **Auth** | None. All data publicly accessible via anonymous S3 (`--no-sign-request`). CC-BY-4.0. |
| **Protocol** | S3 object storage only. No REST/GraphQL API. |
| **S3 buckets** | `s3://janelia-cosem-datasets/` (FIB-SEM + segmentations), `s3://janelia-cosem-networks/` (ML models), `s3://janelia-cosem-publications/` (pub-linked data). Region: us-east-1. |
| **Python clients** | `fibsem-tools` (primary, returns dask-backed xarray), `cellmap-schemas` (CLI for N5/Zarr metadata validation), `cellmap-data` (PyTorch DataLoader integration) |
| **Data formats** | N5 (older), Zarr/OME-NGFF (newer), Neuroglancer Precomputed (web viewing). Multi-TB datasets. |
| **Rate limits** | Standard AWS S3 throughput. No API rate limits (no API). |

**Gaps:**
- **No programmatic dataset discovery** — must `aws s3 ls` to enumerate datasets
- **No search/filter by organism, cell type, organelle** outside web portal
- **No metadata query API** — dataset metadata fragmented across `attributes.json`, Figshare, GitHub READMEs, portal's internal Neo4j (not exposed)
- **This is the repo with the most to gain from an agentic layer** — currently zero programmatic discoverability

---

## 2. API Maturity Comparison

| Capability | EMPIAR | IDR | MICrONS | BIA | OpenOrganelle |
|---|:---:|:---:|:---:|:---:|:---:|
| REST API exists | Yes | Yes (rich) | Yes (rich) | Partial | **No** |
| Search/filter | **No** | Yes (MAPR) | Yes (materialize) | **No** | **No** |
| Entry enumeration | **No** | Yes | Yes | Partial | **No** |
| No auth for reads | Yes | Yes | **No** (token req'd) | Yes | Yes |
| Python client | Partial | Yes | Yes (excellent) | Partial | Partial |
| Cloud-native data | No | Yes (OME-Zarr) | Yes (Precomputed) | Yes (OME-Zarr) | Yes (N5/Zarr) |
| File-level metadata | **No** | Yes | Yes | Partial | **No** |

**Most API-complete:** IDR (richest REST surface, multiple query layers, well-documented)
**Most programmatically complete:** MICrONS/CAVE (excellent Python client, but auth friction)
**Most gap-filled by agentic layer:** OpenOrganelle (zero API), EMPIAR (no search), BIA (fragmented)

---

## 3. Where an Agentic Discovery Layer Adds Value

### Cross-Repo Gaps (what no single repo solves)
1. **No unified search across repos** — each has different query semantics (or none at all)
2. **No cross-referencing** — e.g., "find all FIB-SEM datasets of mitochondria in human cells" requires querying 3+ repos
3. **No format-aware data access** — each repo uses different storage backends (S3, FTP, GCS, OMERO)
4. **No metadata normalization** — each uses different schemas (REMBI, OMERO map annotations, CAVE tables, N5 attributes)

### High-Value Agentic Capabilities
1. **Semantic search over metadata** — natural language to structured queries across all repos
2. **Dataset recommendation** — "find datasets similar to EMPIAR-10310" across repos
3. **Automated data access pattern generation** — given a dataset ID, generate the right Python code to load it (different per repo)
4. **Gap-filling scraping** — for repos without APIs (OpenOrganelle), agent can parse S3 listings and `attributes.json` files to build a searchable index
5. **Cross-reference resolution** — link EMPIAR entries to their IDR counterparts, BIA submissions, or OpenOrganelle datasets

---

## 4. Unified Query Interface Design

```
UnifiedQuery {
  text_query: str              // "FIB-SEM of HeLa cells with segmented mitochondria"
  repositories: [str]          // ["EMPIAR", "IDR", "OpenOrganelle"] or omit for all
  organism: str                // "Homo sapiens"
  cell_type: str               // "HeLa"
  imaging_modality: str        // "FIB-SEM", "cryo-EM", "confocal", "light-sheet"
  data_type: str               // "raw", "segmentation", "reconstruction"
  organelle: str               // "mitochondria", "ER", "nucleus"
  resolution_nm: (float,float) // min/max voxel size
  has_segmentation: bool
  limit: int
  offset: int
}

UnifiedResult {
  repository: str              // source repo
  accession: str               // native ID (EMPIAR-10310, idr0001, minnie65, S-BIAD570, jrc_hela-2)
  title: str
  description: str
  organism: str
  imaging_modality: str
  data_formats: [str]          // ["n5", "zarr", "mrc"]
  access_url: str              // most direct access path
  python_snippet: str          // generated code to load this dataset
  metadata_completeness: float // 0-1 score
  cross_references: [{repo, id}]
}
```

---

## 5. MCP Design Notes (per repo)

### IDR (start here — most API-complete)
- 6 tools: `search_by_gene`, `search_by_organism`, `list_studies`, `get_dataset_images`, `get_image_annotations`, `get_thumbnail_url`
- No auth needed. Pure HTTP GET. Ideal starting point.
- MAPR endpoints enable structured biological queries.

### EMPIAR
- 3 tools: `get_entry`, `get_entries_batch`, `get_by_emdb_id`
- Very thin API — an agent would need to augment with web scraping for search.
- Potential: build an offline index of all entries, expose via semantic search tool.

### MICrONS
- 4 tools: `query_annotations`, `get_cell_types`, `get_synapses`, `list_tables`
- Auth token required — MCP server must handle token management.
- Rich query capabilities via materialization engine.

### BioImage Archive
- 3 tools: `get_study`, `get_study_info`, `list_s3_data`
- Split API surface. BioStudies for metadata, S3 for data access.
- Could unify both under a single tool interface.

### OpenOrganelle
- 3 tools: `list_datasets`, `get_dataset_metadata`, `get_dataset_access_info`
- Must scrape S3 bucket + parse `attributes.json` for metadata.
- Highest value-add for agentic layer — currently zero discoverability.

---

## 6. Key Questions for Anthropic Engineers

### Architecture
1. **Multi-repo fan-out:** Should the MCP server fan out queries to all 5 repos in parallel, or should there be one MCP server per repo with a meta-orchestrator?
2. **Offline index vs. live queries:** For repos without search APIs (EMPIAR, OpenOrganelle), should the agent build/maintain a local metadata index, or scrape on-demand?
3. **Auth delegation:** MICrONS requires token auth. How should an MCP server handle per-user tokens — environment variables, OAuth flow, or token passthrough?

### Agentic Capabilities
4. **Tool composition:** Can the agent chain tools (e.g., search IDR -> get annotations -> cross-reference EMPIAR) autonomously, or does each step need user confirmation?
5. **Data format bridging:** When an agent finds a dataset in N5 but the user needs Zarr, should the MCP server handle conversion or just report the format gap?
6. **Semantic search over metadata:** Should we embed dataset descriptions into a vector store accessible via MCP, or rely on Claude's context window?

### Practical
7. **Rate limiting / caching:** What's the recommended pattern for caching API responses in an MCP server to avoid hammering public repos?
8. **Large result sets:** MICrONS synapses table has 337M rows. How should the MCP server handle queries that could return massive results — streaming, pagination, or server-side aggregation?
9. **Web scraping as fallback:** For OpenOrganelle (no API), is it acceptable for the MCP server to parse S3 bucket listings and `attributes.json` files? What about scraping the web portal?
