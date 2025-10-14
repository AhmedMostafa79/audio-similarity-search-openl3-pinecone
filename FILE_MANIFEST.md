# File Manifest

This manifest enumerates files in the public showcase and their purposes. Private-only components are listed for transparency but not included.

## Public Files
- `README.md` – Project overview & usage
- `requirements.txt` – Python dependencies (minimal demo set)
- `.env.example` – Environment variable template
- `.gitignore` – Ignore large data, secrets, caches
- `FILE_MANIFEST.md` – This document
- `notebooks/` (to add) – Contains `Mini_OpenL3_Demo.ipynb` (sanitized demo)
- `src/` (to add) – Modular Python scripts (preprocessing, pinecone client, utils, viz)
- `data/sample_audio/` (to add) – Few tiny public-domain WAV clips
- `assets/` (to add) – Architecture diagram, t-SNE image

## Private Components (Not in Public Repo)
- Full recitation dataset (large WAV/MP3 files)
- Complete feature arrays (.npy) & combined metadata CSVs
- Checkpointed batch extraction pipeline (large scale)
- Extended evaluation notebooks (noise, website, youtube sets)
- Performance & latency benchmark scripts

## Rationale for Exclusions
Protects proprietary dataset curation effort and prevents raw cloning of full pipeline while still demonstrating competency and architecture.

## How to Reproduce Demo
Provide 3–5 short WAV files in `data/sample_audio/` and run the demo notebook after setting Pinecone API credentials.
