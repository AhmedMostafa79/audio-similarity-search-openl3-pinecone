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

## Evaluation (Notebook-Derived)
The original OpenL3 notebook constructs three held-out test subsets and an aggregated evaluation set:

- `website_test`: Clean recitations (baseline quality)
- `youtube_test`: Compressed / variable quality clips
- `noise_test`: Stratified sample with added environmental/synthetic noise (50 clips per reciter sampled)
- `df_total`: Concatenation of the three above for overall scoring

### Retrieval Metric
Top-K accuracy (K=1 primary; optional K=3) using Pinecone cosine similarity. Each query is one embedding; a hit occurs if the true `name` (reciter) appears in the first K matches (metadata `sheikh_name`).

### Core Evaluation Code (excerpt)
```python
accuracy, true_matches, errors = evaluate_model(X_Total, y_Total, audio_engine, verbose=False)
print({
	'top1_accuracy': accuracy,
	'true_matches': true_matches,
	'errors': errors,
	'total_tests': true_matches + errors
})

def evaluate_topk(X_test, y_test, engine, k=3):
	total=0; hits=0
	for vec, (_, row) in zip(X_test['features'], y_test.iterrows()):
		q = engine.search(vec.tolist(), top_k=k)
		if 'matches' in q:
			total += 1
			names = [m['metadata']['sheikh_name'] for m in q['matches']]
			if row['name'] in names:
				hits += 1
	return hits/total if total else 0.0

top3_accuracy = evaluate_topk(X_Total, y_Total, audio_engine, k=3)
```

### Public Summary Table (fill with your results or keep partial)
| Set          | Clips | Reciters | Top‑1 Acc. | Top‑3 Acc. | Notes |
|--------------|-------|----------|-----------:|-----------:|-------|
| Website      | <FILL>| <FILL>   |  <FILL>%   |  <FILL>%   | Clean baseline |
| YouTube      | <FILL>| <FILL>   |  <FILL>%   |  <FILL>%   | Compression impact |
| Noise (50 ea)| <FILL>| <FILL>   |  <FILL>%   |  <FILL>%   | Noise robustness |
| Combined     | <FILL>| <FILL>   |  <FILL>%   |  <FILL>%   | Overall retrieval |

Latency (optional): measure median per-query time over N random samples.

### Reporting Guidance
- Provide relative robustness deltas (e.g., Noise top‑1 −X% vs Website).
- If protecting IP, publish percentages only, leave absolute clip counts private or bucketed.
- State embedding size (512) & metric (cosine) for reproducibility.

### Privacy Note
Full dataset composition, large NPY feature arrays, and detailed benchmark logs are withheld to prevent trivial cloning while demonstrating methodology.

## Request Full Access
Email <your-email@example.com> or open an issue titled `Access Request` with your affiliation. A temporary private repo invitation will be provided.

## Tech Stack
Python, OpenL3 (TensorFlow), Librosa, Pinecone, NumPy/Pandas, scikit-learn, Matplotlib/Seaborn/Plotly.

## License
MIT for the public showcase. Full pipeline code is All Rights Reserved.

## Disclaimer
Sample data is synthetic / public domain. Real dataset and metrics intentionally withheld to protect IP.
