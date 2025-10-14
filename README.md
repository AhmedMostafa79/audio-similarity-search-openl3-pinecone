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

## Dataset & Reciters
- Source: Majority of audio derived from Tarteel website, which provides ayah-level Quran recitations. We used these to construct our own dataset consisting of per-clip metadata and vector embeddings.
- Reciters (initial 4):
	- Mahmoud Khalil Al-Husary
	- Abdulrahman Al-Sudais
	- Maher Al-Muaiqly
	- Yasser Al-Dusary
- Composition (approximate):
	- Clean ayah clips: ~1,100 per reciter (≈4,400 total)
	- AI-generated noised clips: ~100 per reciter (≈400 total)
	- Manually recorded real-time clips: ~40 per reciter (≈160 total)
	- Grand total: ≈4,960 clips (≈1,240 per reciter)

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

## Results
Metric: Top‑1 retrieval accuracy via cosine similarity in Pinecone (correct if the top match’s reciter equals ground truth).

- Combined verse retrieval: 94.64% accuracy (1,129 correct / 64 wrong), N = 1,193 queries.

Evaluation image:

![Evaluation summary](assets/evaluation_summary.jpg)

Privacy note: Full feature arrays and detailed logs remain private; we publish aggregate metrics and representative visuals.

## Request Full Access
Email <ahmedkhater7779@gmail.com> or open an issue titled `Access Request` with your affiliation. A temporary private repo invitation will be provided.

## Tech Stack
Python, OpenL3 (TensorFlow), Librosa, Pinecone, NumPy/Pandas, scikit-learn, Matplotlib/Seaborn/Plotly.

## License
MIT for the public showcase. Full pipeline code is All Rights Reserved.

## Disclaimer
Sample data is synthetic / public domain. Real dataset and metrics intentionally withheld to protect IP.
