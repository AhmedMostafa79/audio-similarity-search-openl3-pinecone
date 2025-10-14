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


## Evaluation (Derived From Notebook Pipeline)
This project evaluates retrieval accuracy across three held‑out subsets built in the notebook:

1. `website_test`: Clean recitations (baseline quality)
2. `youtube_test`: Compressed / variable quality streams
3. `noise_test`: Stratified sample (50 clips per reciter) with added environmental / synthetic noise

An aggregated frame `df_total` concatenates all three for overall scoring, and `X_Total / y_Total` feed the retrieval evaluation function.

### Retrieval Metric
Top‑K Accuracy (primarily K=1, optionally K=3) is computed by querying Pinecone with each embedding and checking if the true reciter appears among the first K matches (`metadata['sheikh_name']`).

### Core Evaluation Code (simplified)
```python
accuracy, true_matches, errors = evaluate_model(X_Total, y_Total, audio_engine, verbose=False)
print({
	'top1_accuracy': accuracy,
	'true_matches': true_matches,
	'errors': errors,
	'total_tests': true_matches + errors
})
```

To extend to Top‑3:
```python
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

### Suggested Public Summary Table (replace <FILL> with your actual counts)
| Set          | Clips | Reciters | Top‑1 Acc. | Top‑3 Acc. | Notes |
|--------------|-------|----------|-----------:|-----------:|-------|
| Website      | <FILL>| <FILL>   |  <FILL>%   |  <FILL>%   | Clean baseline |
| YouTube      | <FILL>| <FILL>   |  <FILL>%   |  <FILL>%   | Compression impact |
| Noise (50 ea)| <FILL>| <FILL>   |  <FILL>%   |  <FILL>%   | Added noise stress |
| Combined     | <FILL>| <FILL>   |  <FILL>%   |  <FILL>%   | Overall retrieval |

Latency (optional): measure median query time by timing `engine.search` over N random samples.

### Reporting Guidance
- Include relative robustness: e.g., `Noise top‑1 −4.2% vs Website`.
- If privacy sensitive, publish percentages only; keep raw clip counts private or bucket (e.g., 500–1K).
- State index dimension (512) and metric (cosine) for reproducibility.

### Privacy Rationale
Exact large dataset makeup, full embeddings and full evaluation logs are withheld to reduce cloning risk while still demonstrating methodology and reproducibility steps.


## Request Full Access
Email <your-email@example.com> or open an issue titled `Access Request` with your affiliation. A temporary private repo invitation will be provided.

## Tech Stack
Python, OpenL3 (TensorFlow), Librosa, Pinecone, NumPy/Pandas, scikit-learn, Matplotlib/Seaborn/Plotly.

## License
MIT for the public showcase. Full pipeline code is All Rights Reserved.

## Disclaimer
Sample data is synthetic / public domain. Real dataset and metrics intentionally withheld to protect IP.
