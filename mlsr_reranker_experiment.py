import os
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
import pyterrier as pt
from sentence_transformers import SentenceTransformer, SparseEncoder, util

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

PROJECT_ROOT = Path("/Users/user/Submissions/BEA-2026").resolve()
QB_PATH = PROJECT_ROOT / "qbank.csv"
QUERIES_PATH = PROJECT_ROOT / "queries.csv"
QRELS_PATH = PROJECT_ROOT / "qrels.tsv"

ART_DIR = PROJECT_ROOT / "artifacts" / "mlsr_reranker"
ART_DIR.mkdir(parents=True, exist_ok=True)

DENSE_MODEL = "deutsche-telekom/gbert-large-paraphrase-cosine"
SPARSE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
DENSE_TOPK = 100
FINAL_TOPK = 50
BATCH_SIZE = 16

_ws_re = re.compile(r"\s+")
_tags_re = re.compile(r"<[^>]+>")


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = _tags_re.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[^0-9a-zA-ZäöüßÄÖÜẞ]+", " ", s)
    s = _ws_re.sub(" ", s).strip()
    return s


def safe_concat(parts):
    clean = []
    for p in parts:
        if p is None:
            continue
        p = str(p)
        if p.strip() in ("", "N/A", "nan"):
            continue
        clean.append(p)
    return " ".join(clean)


def build_doc_text(row: pd.Series) -> str:
    return safe_concat([row.get("question"), row.get("choices_processed")])


def ensure_run(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "qid" not in df.columns or "docno" not in df.columns:
        raise ValueError("Run must contain columns: qid, docno")
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    if "score" not in df.columns:
        df["score"] = 0.0
    df["score"] = df["score"].astype(float)
    if "rank" not in df.columns:
        df = df.sort_values(["qid", "score"], ascending=[True, False])
        df["rank"] = df.groupby("qid").cumcount() + 1
    return df


if not pt.java.started():
    pt.java.init()


def get_or_build_index(index_dir: Path, corpus_df: pd.DataFrame) -> pt.IndexRef:
    index_dir = Path(index_dir)
    if (index_dir / "data.properties").exists():
        return pt.IndexRef.of(str(index_dir))
    index_dir.mkdir(parents=True, exist_ok=True)
    indexer = pt.IterDictIndexer(str(index_dir), meta={"docno": 64}, text_attrs=["text"])
    return indexer.index(corpus_df[["docno", "text"]].to_dict("records"))


class FaissDenseRetriever(pt.Transformer):
    def __init__(self, corpus_df: pd.DataFrame, model_name: str, topk: int, show_progress: bool = True):
        super().__init__()
        self.topk = int(topk)
        cdf = corpus_df[["docno", "text"]].copy()
        cdf["docno"] = cdf["docno"].astype(str)
        cdf["text"] = cdf["text"].astype(str)
        self.docnos = cdf["docno"].tolist()
        self.st = SentenceTransformer(model_name, device="cpu")
        xdoc = self.st.encode(
            cdf["text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=bool(show_progress),
        ).astype("float32")
        self.index = faiss.IndexFlatIP(xdoc.shape[1])
        self.index.add(xdoc)

    def transform(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        qids = topics_df["qid"].astype(str).tolist()
        qs = topics_df["query"].astype(str).tolist()
        q = self.st.encode(qs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        scores, idxs = self.index.search(q, self.topk)
        rows = []
        for i, qid in enumerate(qids):
            for rank, (j, sc) in enumerate(zip(idxs[i], scores[i]), start=1):
                if j < 0:
                    continue
                rows.append({"qid": qid, "docno": self.docnos[j], "score": float(sc), "rank": int(rank)})
        return pd.DataFrame(rows)


class SparseEncoderRetriever(pt.Transformer):
    def __init__(self, corpus_df: pd.DataFrame, model_name: str, topk: int = 50, batch_size: int = 16):
        super().__init__()
        cdf = corpus_df[["docno", "text"]].copy()
        cdf["docno"] = cdf["docno"].astype(str)
        cdf["text"] = cdf["text"].astype(str)
        self.docnos = cdf["docno"].tolist()
        self.topk = int(topk)
        self.batch_size = int(batch_size)
        self.model = SparseEncoder(model_name, device="cpu")
        self.doc_emb = self.model.encode_document(
            cdf["text"].tolist(),
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

    def transform(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        qids = topics_df["qid"].astype(str).tolist()
        qs = topics_df["query"].astype(str).tolist()
        query_emb = self.model.encode_query(
            qs,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        scores = util.dot_score(query_emb, self.doc_emb)
        rows = []
        for qi, qid in enumerate(qids):
            top = scores[qi].topk(k=self.topk)
            for rank, (score, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), start=1):
                rows.append({"qid": qid, "docno": self.docnos[int(idx)], "score": float(score), "rank": int(rank)})
        return pd.DataFrame(rows)


class SparseEncoderReranker(pt.Transformer):
    def __init__(self, corpus_df: pd.DataFrame, model_name: str, batch_size: int = 16):
        super().__init__()
        cdf = corpus_df[["docno", "text"]].copy()
        cdf["docno"] = cdf["docno"].astype(str)
        cdf["text"] = cdf["text"].astype(str)
        self.docno2text = dict(zip(cdf["docno"], cdf["text"]))
        self.model = SparseEncoder(model_name, device="cpu")
        self.batch_size = int(batch_size)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = ensure_run(df)
        if "query" not in df.columns:
            raise ValueError("SparseEncoderReranker needs query column, merge topics first")
        out_groups = []
        for qid, g in df.groupby("qid", sort=False):
            qtext = str(g["query"].iloc[0])
            docnos = g["docno"].astype(str).tolist()
            doc_texts = [self.docno2text.get(docno, "") for docno in docnos]
            q_emb = self.model.encode_query(
                [qtext],
                batch_size=1,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            d_emb = self.model.encode_document(
                doc_texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            scores = util.dot_score(q_emb, d_emb).squeeze(0).detach().cpu().numpy().astype(float)
            gg = g.copy()
            gg["score"] = scores
            gg = gg.sort_values("score", ascending=False)
            gg["rank"] = np.arange(1, len(gg) + 1)
            out_groups.append(gg)
        return pd.concat(out_groups, ignore_index=True)


def cut_k(k: int) -> pt.Transformer:
    return pt.apply.generic(lambda df: df.groupby("qid", sort=False).head(int(k)))


def add_query_col(topics_df: pd.DataFrame) -> pt.Transformer:
    tq = topics_df[["qid", "query"]].copy()
    tq["qid"] = tq["qid"].astype(str)

    def _merge(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["qid"] = df["qid"].astype(str)
        return df.merge(tq, on="qid", how="left")

    return pt.apply.generic(_merge)


def eval_at_k(pipelines: List[Tuple[str, pt.Transformer]], topics_df: pd.DataFrame, qrels_df: pd.DataFrame, k_eval: int) -> pd.DataFrame:
    metrics = [
        f"ndcg_cut.{k_eval}",
        "recip_rank",
        f"P.{k_eval}",
        f"recall.{k_eval}",
        f"map_cut.{k_eval}",
    ]
    df = pt.Experiment(
        [p for _, p in pipelines],
        topics_df,
        qrels_df,
        eval_metrics=metrics,
        names=[n for n, _ in pipelines],
        verbose=True,
        validate="ignore",
    )
    df = df.rename(columns={
        f"ndcg_cut.{k_eval}": f"nDCG@{k_eval}",
        "recip_rank": f"MRR@{k_eval}",
        f"P.{k_eval}": f"Prec@{k_eval}",
        f"recall.{k_eval}": f"Recall@{k_eval}",
        f"map_cut.{k_eval}": f"MAP@{k_eval}",
    }).copy()
    p = df[f"Prec@{k_eval}"].astype(float)
    r = df[f"Recall@{k_eval}"].astype(float)
    df[f"F1@{k_eval}"] = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    cols = ["name", f"nDCG@{k_eval}", f"MRR@{k_eval}", f"Prec@{k_eval}", f"Recall@{k_eval}", f"F1@{k_eval}", f"MAP@{k_eval}"]
    return df[cols]


def to_table1_style(metrics_df: pd.DataFrame, ann_name: str, k_eval: int = 50) -> pd.DataFrame:
    df = metrics_df.copy()
    ann_ndcg = float(df.loc[df["name"] == ann_name, f"nDCG@{k_eval}"].iloc[0])
    df["Delta"] = df[f"nDCG@{k_eval}"].astype(float) - ann_ndcg
    df["Pct"] = np.where(ann_ndcg != 0, 100.0 * df["Delta"] / ann_ndcg, 0.0)
    out = pd.DataFrame({
        "Method": df["name"],
        "nDCG": df[f"nDCG@{k_eval}"].astype(float),
        "Delta": df["Delta"].astype(float),
        "%": df["Pct"].astype(float),
        "MRR": df[f"MRR@{k_eval}"].astype(float),
        "P": df[f"Prec@{k_eval}"].astype(float),
        "R": df[f"Recall@{k_eval}"].astype(float),
        "F1": df[f"F1@{k_eval}"].astype(float),
        "MAP": df[f"MAP@{k_eval}"].astype(float),
    })
    return out


def main() -> None:
    qb = pd.read_csv(QB_PATH).fillna("N/A")
    queries = pd.read_csv(QUERIES_PATH).fillna("")
    qrels = pd.read_csv(QRELS_PATH, sep="\t").fillna(0)

    qb["docno"] = qb["test_item_id"].astype(str)
    qb["raw_text"] = qb.apply(build_doc_text, axis=1)
    qb["text"] = qb["raw_text"].map(normalize_text)
    corpus = qb[["docno", "text"]].copy()

    topics = queries.rename(columns={"queries": "query"})[["qid", "query"]].copy()
    topics["qid"] = topics["qid"].astype(str)
    topics["query"] = topics["query"].astype(str).map(normalize_text)

    qrels = qrels.rename(columns={"rel": "label"})[["qid", "docno", "label"]].copy()
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)
    qrels["label"] = qrels["label"].astype(int)

    index_ref = get_or_build_index(ART_DIR / "terrier_index", corpus)
    dense100 = FaissDenseRetriever(corpus, model_name=DENSE_MODEL, topk=DENSE_TOPK)
    sparse50 = SparseEncoderRetriever(corpus, model_name=SPARSE_MODEL, topk=FINAL_TOPK, batch_size=BATCH_SIZE)
    sparse_rerank = SparseEncoderReranker(corpus, model_name=SPARSE_MODEL, batch_size=BATCH_SIZE)
    cut50 = cut_k(FINAL_TOPK)
    add_query = add_query_col(topics)
    lex_bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")

    pipelines: List[Tuple[str, pt.Transformer]] = [
        ("ANN@100->@50", dense100 >> cut50),
        ("dense@100>>BM25(rescore)->@50", dense100 >> add_query >> lex_bm25 >> cut50),
        ("MLSR->@50", sparse50),
        ("dense@100>>MLSR(rescore)->@50", dense100 >> add_query >> sparse_rerank >> cut50),
    ]

    metrics_df = eval_at_k(pipelines, topics, qrels, k_eval=FINAL_TOPK)
    metrics_df = metrics_df.sort_values(f"nDCG@{FINAL_TOPK}", ascending=False).reset_index(drop=True)
    table_df = to_table1_style(metrics_df, ann_name="ANN@100->@50", k_eval=FINAL_TOPK)
    table_df = table_df.sort_values("nDCG", ascending=False).reset_index(drop=True)

    metrics_path = ART_DIR / "mlsr_reranker_metrics_raw.csv"
    table_path = ART_DIR / "mlsr_reranker_table1_style.csv"

    metrics_df.to_csv(metrics_path, index=False)
    table_df.to_csv(table_path, index=False)

    print(metrics_df.to_string(index=False))
    print()
    print(table_df.to_string(index=False))
    print()
    print("Saved:", metrics_path)
    print("Saved:", table_path)


if __name__ == "__main__":
    main()
