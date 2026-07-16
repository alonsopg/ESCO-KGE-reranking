# ESCO-KGE-reranking

Code and experiment artifacts for the BEA 2026 paper on ESCO-grounded
knowledge-graph re-ranking for German vocational education and training
(VET) item retrieval.

## Repository layout

- `qbank.csv`, `queries.csv`, `qrels.tsv`: item bank, retrieval topics, and
  relevance judgments used by the experiments.
- `ESCO/`: German ESCO release files used to build skill links and graph
  triples, including `skills_de.csv`, `skillSkillRelations_de.csv`, and
  `skillsHierarchy_de.csv`.
- `experiments-2026.ipynb`: original KG re-ranking experiments with dense,
  lexical, cross-encoder, ComplEx, RotatE, and initial SVD variants.
- `svd_simonly_512_*.ipynb`: focused SVD reranker experiments and link/alpha
  sweeps for the strongest ESCO-only setting.
- `ESCO-linking-diagnostics.ipynb`: ESCO linking coverage, threshold checks,
  and example links.
- `bge_gemma_reranker_experiment.ipynb`: BGEGemma reranker baseline.
- `kg_bge_hybrid_fusion_experiment.ipynb`: late-fusion KG+BGEGemma hybrid.
- `mlsr_reranker_experiment.py`: multilingual learned sparse retriever and
  re-ranker baseline.
- `artifacts/`: exported metrics, per-query scores, plots, and summary files
  used to report the paper results.

## Environment

The experiments were run with Python 3.10/3.11. PyTerrier also requires Java.
The notebooks and script use the following main packages:

```bash
pip install pandas numpy scipy scikit-learn matplotlib jupyter
pip install torch sentence-transformers transformers accelerate
pip install faiss-cpu python-terrier pykeen ranx
```

GPU or Apple MPS acceleration is useful for the dense and BGEGemma rerankers,
but the code falls back to CPU where supported.

## Data and paths

Most notebooks define a `PROJECT_ROOT` cell near the top. When running outside
the original machine, update that cell so it points to the cloned repository.
For ESCO-based runs, ensure these files exist:

```text
ESCO/skills_de.csv
ESCO/skillSkillRelations_de.csv
ESCO/skillsHierarchy_de.csv
qbank.csv
queries.csv
qrels.tsv
```

Some older notebook cells contain absolute paths from earlier exploratory
runs; the later focused notebooks use the repository-local `ESCO/` directory
and write outputs under `artifacts/`.

## Headline experimental settings

- Initial dense retrieval: `deutsche-telekom/gbert-large-paraphrase-cosine`,
  top-100 candidates.
- ESCO skill linking: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
- Link pruning used in the paper: query top-20, document top-15, cosine
  similarity `>= 0.45`.
- Main KG reranker: SVD sim-only skill representation, 512 dimensions,
  direct similarity-weighted ESCO links, interpolation `alpha = 0.30`.
- BGEGemma baseline: `BAAI/bge-reranker-v2-gemma`, zero-shot re-ranking of the
  dense top-100 candidate set, maximum sequence length 512.
- Cross-encoder baseline: `cross-encoder/msmarco-MiniLM-L6-en-de-v1`,
  zero-shot, without VET-specific fine-tuning.
- Multilingual learned sparse baseline:
  `opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1`.
- Best late-fusion hybrid: SVD KG reranker plus BGEGemma with `beta = 0.55`.

## Suggested reproduction order

1. Run `ESCO-linking-diagnostics.ipynb` to verify ESCO linking coverage and the
   `0.45` pruning threshold diagnostics.
2. Run `experiments-2026.ipynb` for the original dense, lexical, cross-encoder,
   ComplEx, RotatE, and initial SVD comparisons.
3. Run `svd_simonly_512_experiment.ipynb` and
   `svd_simonly_512_alpha_sweep.ipynb` for the strongest standalone KG setting.
4. Run `svd_rank_weighting_ablation.ipynb` or generate it with
   `make_svd_rank_weighting_ablation_notebook.py` for the SVD rank/weighting
   ablation.
5. Run `bge_gemma_reranker_experiment.ipynb` for the main neural re-ranker
   baseline.
6. Run `kg_bge_hybrid_fusion_experiment.ipynb` for the KG+BGEGemma fusion
   result and significance summaries.
7. Run `python mlsr_reranker_experiment.py` for the multilingual learned sparse
   baseline.

## Main artifacts

- `artifacts/linking_analysis/`: ESCO linking statistics and example links.
- `artifacts/graph_sparsity/`: KG sparsity summary.
- `artifacts/svd_ablation/`: SVD rank and weighting ablation outputs.
- `artifacts/svd_simonly_512_alpha_sweep/`: alpha sweep for the best SVD
  setting.
- `artifacts/bge_gemma_reranker/`: BGEGemma baseline metrics.
- `artifacts/kg_bge_hybrid/`: late-fusion KG+BGEGemma metrics, per-query
  scores, plots, and significance tests.
- `artifacts/mlsr_reranker/`: learned sparse baseline outputs.

The key reported summary files are:

```text
artifacts/linking_analysis/linking_summary_paragraph.txt
artifacts/graph_sparsity/kg_sparsity_summary.txt
artifacts/svd_ablation/svd_rank_weighting_summary.txt
artifacts/svd_simonly_512_alpha_sweep/svd_simonly_512_alpha_sweep_summary.txt
artifacts/kg_bge_hybrid/kg_bge_hybrid_summary.txt
artifacts/kg_bge_hybrid/kg_bge_hybrid_significance_summary.txt
```

## Current headline results

- Best standalone ESCO KG reranker: nDCG@50 = 0.6468.
- BGEGemma reranker baseline: nDCG@50 = 0.6008.
- Best KG+BGEGemma hybrid: nDCG@50 = 0.6700.

For exact values and statistical tests, use the exported CSV and TXT files in
`artifacts/`.

## Citation

If you use this code, please cite the accompanying BEA 2026 paper.
