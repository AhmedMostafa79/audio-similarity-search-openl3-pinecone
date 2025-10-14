# Audio Similarity Search (OpenL3 + Pinecone)

Public, sanitized showcase of an audio embedding and similarity pipeline. The full production notebook (batch preprocessing, expanded evaluation sets, additional reciter analytics) remains private and is available to recruiters on request.

## Overview
Pipeline: preprocess small sample audio -> extract 512-D OpenL3 embeddings -> build Pinecone vector index -> cosine top-K search -> simple evaluation + visualization (t-SNE).

## Features (Public Demo)
- Minimal `Preprocess` class (trim, noise-reduce, band-pass, normalize)
- OpenL3 embedding extraction (mel256, music, 512-D)
- Pinecone index creation & vector upsert
- Query API to fetch nearest reciter match
- Basic evaluation accuracy on moderate sample set
- t-SNE & similarity visualizations (static images or generated on demand)

## What Is Private (Full Version)
- Large curated recitation dataset & preprocessing scripts
- Full checkpointed batch feature extraction pipeline
- Extended evaluation suites (website / youtube / noise robustness sets)
- Detailed analytics notebooks & performance metrics

## Quick Start
1. Clone repo
2. Create env & install: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill Pinecone fields
4. Run the demo notebook in `notebooks/` (to be added) or scripts in `src/`
5. (Optional) Generate t-SNE visualization using provided helper

## Demo
| Resource | Link |
|----------|------|
| Video (2‑min overview) | <ADD_UNLISTED_YOUTUBE_OR_LOOM_LINK> |
| Architecture Diagram | assets/architecture.png |
| t-SNE Screenshot | assets/tsne.png |

Add a short unlisted YouTube or Loom video: intro (10s) → architecture (30s) → notebook run (40s) → Pinecone query response (20s) → closing (10s). Keep under 2 minutes.

## Resume Snippet
```
Audio Similarity Search (OpenL3 + Pinecone) – Built pipeline: preprocess → 512‑D OpenL3 embeddings → Pinecone vector index → cosine top‑K search + t‑SNE viz. Public mini repo & demo; full dataset + evaluation private (available on request).
```


## Evaluation (Demo Subset)
The public demo intentionally uses a tiny synthetic/public sample. Metrics below are illustrative placeholders—replace with real numbers from your private evaluation.

| Set              | Samples (clips) | Unique Reciters | Top‑1 Accuracy | Top‑3 Accuracy | Notes |
|------------------|-----------------|-----------------|----------------|----------------|-------|
| Demo Train Index |  <FILL>         |  <FILL>         |  —             |  —             | Used only to build small index |
| Website Test     |  <FILL>         |  <FILL>         |  <FILL>%       |  <FILL>%       | Clean recordings subset |
| YouTube Test     |  <FILL>         |  <FILL>         |  <FILL>%       |  <FILL>%       | Mixed compression quality |
| Noise Stress     |  <FILL>         |  <FILL>         |  <FILL>%       |  <FILL>%       | Added synthetic/environmental noise |
| Combined Total   |  <FILL>         |  <FILL>         |  <FILL>%       |  <FILL>%       | Consolidated evaluation |

Reporting guidance:
- Define Top‑K Accuracy: fraction of queries where true reciter appears in top K Pinecone results.
- Provide latency (median query ms) if measured on commodity hardware.
- Optionally include robustness deltas (e.g., Noise set: −X% vs clean).

Privacy rationale: Full dataset sizes, full embeddings, and aggregate benchmarks remain private to prevent trivial cloning of production pipeline.


## Request Full Access
Email <your-email@example.com> or open an issue titled `Access Request` with your affiliation. A temporary private repo invitation will be provided.

## Tech Stack
Python, OpenL3 (TensorFlow), Librosa, Pinecone, NumPy/Pandas, scikit-learn, Matplotlib/Seaborn/Plotly.

## License
MIT for the public showcase. Full pipeline code is All Rights Reserved.

## Disclaimer
Sample data is synthetic / public domain. Real dataset and metrics intentionally withheld to protect IP.
