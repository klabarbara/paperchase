import pandas as pd, json, pathlib
from tqdm import tqdm
from pathlib import Path
from pyserini.index.lucene import LuceneIndexer

CSV_PATH  = pathlib.Path("data/processed/papersum_clean.csv")
JSONL_DIR = pathlib.Path("data/processed/jsonl"); JSONL_DIR.mkdir(exist_ok=True)

cols = ["paperID","title","abstract","venue","year","conclusion"]

INDEX_DIR = Path("index/bm25")

JSONL_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# some abstracts are not present. treat NaN as empty string so lucene can process it
def safe(val):
    if val is None:
        return ""
    try:
        import math, numpy as np
        if isinstance(val, (float, np.floating)) and math.isnan(val):
            return ""
    except Exception:
        pass
    return val

def csv_to_jsonl():
    print("Converting CSV to JSONL files...")

    try:
        for chunk in tqdm(
                pd.read_csv(CSV_PATH, usecols=cols, chunksize=50_000, dtype=str),
                desc="Converting to JSONL"
            ):
            # build the 'text' column in one shot
            chunk["text"] = chunk["title"] + ". " + chunk["abstract"].fillna("") + \
                            " " + chunk["conclusion"].fillna("")
            for rec in chunk.to_dict(orient="records"):
                out = JSONL_DIR / f"{rec['paperID']}.json"
                out.write_text(json.dumps({
                    "id":        rec["paperID"],
                    "title":     rec["title"],
                    "abstract":  safe(rec["abstract"]),
                    "venue":     rec["venue"],
                    "year":      rec["year"],
                    "text":      rec["text"],
                    "contents":  f"{rec['title']} {rec['abstract']} {rec['text']}"
                }, allow_nan=False), encoding="utf‑8")

    except Exception as e:
        print(f"error during csv to jsonl conversion: {e}")

    print(f"CSV to JSONL conversion completed. Files are stored in {JSONL_DIR}")

def build_bm25():
    print("Building BMN25 Index with Lucene Indexer...")
    writer = LuceneIndexer(str(INDEX_DIR))

    json_files = list(JSONL_DIR.glob("*.json"))

    if not json_files:
        print(f"no json files found in {JSONL_DIR}. exiting")
        return

    for path in tqdm(json_files, desc="Indexing Documents"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            writer.add_doc_raw(content)
        except Exception as e:
            print(f"error indexing {path.name}: {e}")

    writer.commit()
    print(f"Indexing completed. Index is stored in {INDEX_DIR}")

if __name__ == "__main__":
    
    csv_to_jsonl()
    # build_bm25()
            