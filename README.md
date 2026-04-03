# Chapter Reconstruction from Source Notebooks

## Table of Contents
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [How to Run](#how-to-run)
  - [CLI Arguments](#cli-arguments)
  - [Available Themes](#available-themes)
  - [Example Input and Output](#example-input-and-output)
- [Design Decisions](#design-decisions)
  - [Architecture](#architecture)
  - [Why Embeddings?](#why-embeddings)
  - [Why a Multi-Step Generation Pipeline?](#why-a-multi-step-generation-pipeline)
  - [Why Sentence-Transformers for Selection?](#why-sentence-transformers-for-selection)
  - [Why Gemini for Generation?](#why-gemini-for-generation)
  - [Tools and Models Used](#tools-and-models-used)
- [Tradeoffs and Limitations](#tradeoffs-and-limitations)
- [Scaling for Daily Production Runs](#scaling-for-daily-production-runs)
  - [Reducing Cost](#reducing-cost)
  - [Improving Latency and Reliability](#improving-latency-and-reliability)
  - [Architectural Changes](#architectural-changes)

## Setup Instructions

### Prerequisites
- Python 3.10+
- Git (for cloning the source notebook repository)
- A [Google Gemini API key](https://aistudio.google.com/apikey)
  - You can obtain a free one by starting a Google AI Project

### Installation

1. Clone this repository:
```bash
git clone https://github.com/xiezoey/Chapter-Reconstruction-from-Source-Notebooks.git
cd Chapter-Reconstruction-from-Source-Notebooks
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
> **Note:** This may take a few minutes

4. Set your API key in an .env file:
```bash
export GEMINI_API_KEY="your-key-here"
```

## How to Run

```bash
# Auto-clone repo, interactively pick a theme
python main.py

# List available themes
python main.py --list-themes

# Full example with all options
python main.py --theme 0 --top-notebooks 4 --output output/chapter.md
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--repo` | *(auto-clones)* | Path to a local `handson-ml3` clone. If omitted, clones to a temp directory. |
| `--provider` | `gemini` | Only one LLM provider (`gemini`) right now |
| `--model` | `gemini-3.1-flash-lite-preview` | Model name to use |
| `--theme` | *(interactive)* | Theme index (0-3). If omitted, prompts for selection. |
| `--top-notebooks` | `3` | Number of top-scoring notebooks to select |
| `--output` | `output/chapter.md` | Output file path (auto-increments if file exists) |
| `--list-themes` | — | Print available themes and exit |

### Available Themes

| Index | Theme |
|---|---|
| 0 | Supervised Learning Fundamentals: From Linear Models to Ensembles |
| 1 | Deep Learning with Neural Networks: Architecture to Training |
| 2 | The Complete ML Pipeline: From Data to Deployment |
| 3 | Classification: Algorithms, Metrics, and Real-World Applications |

### Example Input and Output

**Input:** Theme 3 (Classification)
```bash
python main.py --theme 3
```

**Selected notebooks:** `03_classification.ipynb`, `05_support_vector_machines.ipynb`, `07_ensemble_learning_and_random_forests.ipynb`

**Output:** See [output/example_output.md](output/example_output.md) for a full generated chapter.

## Design Decisions

### Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Notebooks  │────>│  Stage 1: Ingest │────>│ Stage 2: Select  │────>│  Stage 3:    │
│  (.ipynb)   │     │  (JSON parser)   │     │ (sentence-       │     │  Generate    │
└─────────────┘     └──────────────────┘     │  transformers)   │     │  (Gemini)    │
                                             └──────────────────┘     └──────┬───────┘
                                                                             │
                                                                     ┌───────▼───────┐
                                                                     │  chapter.md   │
                                                                     └───────────────┘
```

1. **Ingest** ([ingest.py](ingest.py)) — Parses `.ipynb` files into structured `NotebookContent` objects. Extracts markdown/code cells and strips boilerplate (imports, magic commands, setup cells) and exercise sections.

2. **Select** ([selector.py](selector.py)) — Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to compute semantic similarity between chapter themes and notebook content. Notebooks are ranked by cosine similarity to the theme embedding, and the top N (default: 3) are chosen. Within those notebooks, individual cells are also scored and filtered by a relevance threshold.

3. **Generate** ([generate.py](generate.py)) — LLM pipeline using Google Gemini:
   1. Generate an **outline** following the one specified in the assignment document
   2. Generate each **section** independently, with the outline as shared context
   3. **Assemble** the sections into a chapter with a generated table of contents

### Design Decisions

#### Why Embeddings?

The selection stage needed to match notebook content to chapter themes, and there were a few possible approaches: keyword matching, sending everything to an LLM, or using embeddings. Keyword matching is fast but doesn't take into account semantic meaning. Sending all notebook cells to an LLM for relevance scoring would work semantically, but would be slow and expensive given the volume of cells across all notebooks (and I don't have many free credits...). Embeddings are able to capture semantic meaning  while running locally with no API cost or rate limits. 

#### Why a Multi-Step Generation Pipeline?

Rather than dumping all source material into a single prompt, the tool chains multiple prompts together, giving each section specific information they need. This produces more structured, coherent output than a single-shot approach and stays within token limits.

#### Why Sentence-Transformers for Selection?

The `sentence-transformers` library was chosen because it runs entirely locally and is free/open-sourced. The specific model, `all-MiniLM-L6-v2`, is fast enough to embed all notebook cells in seconds, while still producing high-quality semantic embeddings. Larger models like `all-mpnet-base-v2` would offer slightly better accuracy but at 3x the size and slower inference (70s and 120mb vs. 200s and 420mb).

#### Why Gemini for Generation?

Google Gemini was chosen mainly because it offers a free API tier. The `gemini-3.1-flash-lite-preview` model provides a good balance of output quality and speed for chapter-length generation tasks, while its generous context window comfortably handles the source material from multiple notebooks. The `create_backend` factory in `generate.py` is designed so that additional providers could be added in the future.

### Tools and Models Used

- **sentence-transformers** (`all-MiniLM-L6-v2`): semantic embedding for notebook/section selection
- **Google Gemini** (`gemini-3.1-flash-lite-preview`): chapter generation via the `google-genai` SDK
- **numpy**: cosine similarity computation
- **python-dotenv**: environment variable management
- **Claude Code**: used as an AI coding assistant during development

## Tradeoffs and Limitations

- **Fixed theme set:** The tool uses 4 pre-defined themes and descriptions rather than allowing the user to choose any theme and generate a description. This keeps selection deterministic and avoids an extra LLM call, but limits flexibility.
- **Truncation of source material:** Long notebooks are truncated to fit within LLM context windows (80K chars for preparation, 15K for outline, 20K for non-core sections). Some relevant content may be lost.
- **No cross-section consistency enforcement:** Each section is generated independently. While the shared outline provides coherence, some overlap between sections is possible.
- **Single LLM provider:** Currently only supports Google Gemini. The `create_backend` factory is designed for extensibility but only one provider is implemented.

## Scaling for Daily Production Runs

### Reducing Cost
- **Cache embeddings:** Notebook embeddings only change when the notebook content changes. Store embeddings keyed by a content hash and recompute only when notebooks are modified.
- **Cache generated sections:** If the same source material is selected for a theme, reuse previously generated sections rather than re-generating.
- **Use cheaper models for outline generation:** The outline step is less quality-sensitive than section writing — a smaller/cheaper model could handle it.

### Improving Latency and Reliability
- **Parallelize section generation:** The 6 sections are independent once the outline exists. Generate them concurrently instead of sequentially.
- **Add retry logic with exponential backoff:** Already partially implemented for rate limits and 503s, but could be more robust with circuit breakers and fallback models.
- **Pre-clone and index the repository:** Instead of cloning on each run, maintain a persistent local mirror updated via cron/webhook.

### Architectural Changes
- **Move to an async pipeline:** Use `asyncio` or a task queue (Celery) to parallelize ingestion, embedding, and generation stages.
- **Add a content-addressed cache layer:** Hash notebook content and theme parameters to create cache keys. Skip recomputation when inputs haven't changed.
- **Decouple stages with a message queue:** Run ingest, select, and generate as independent services that communicate via a queue, enabling horizontal scaling and independent deployment.
- **Implement incremental updates:** Track which notebooks changed since the last run and only re-ingest and re-embed those, merging with cached results for unchanged notebooks.
