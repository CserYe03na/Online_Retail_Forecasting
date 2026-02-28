import os
import json
import time
import re
from typing import List, Dict
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

INPUT_XLSX = "data/online_retail_II_cleaned.xlsx"
SHEETS = ["Year 2009-2010", "Year 2010-2011"]

MODEL = "gpt-4o-mini"
TEMPERATURE = 0
BATCH_SIZE = 30
SLEEP_SEC = 0.05
CHECKPOINT_JSONL = "desc2fam_checkpoint.jsonl"

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

COLOR_WORDS = {
    "RED","GREEN","BLUE","PINK","WHITE","BLACK","SILVER","GOLD","IVORY","CREAM","GREY","GRAY",
    "PURPLE","YELLOW","ORANGE","BROWN","BEIGE","NAVY","TEAL","LILAC","TURQUOISE","CLEAR",
    "LIGHT","DARK"
}

SYSTEM_INSTRUCTIONS = """You are grouping retail product descriptions into product families with "just right" granularity.

Goal:
- The family_name should represent the HEAD NOUN / product type. If a clear product type word appears (often at the end), family_name must be that type (+ optional size only).
- Group true variants (color, flavor, theme, personal names, minor motif) under the same family.
- All materials/themes/Scents/flavors/spices must go to variant_hint, not family_name.
- Do NOT over-merge different base products into one family (e.g., do not put all DOLL items into one family; keep a distinguishing modifier such as material/brand/style if present: FELTCRAFT DOLL vs RAG DOLL).
- Do NOT over-split by colors/names.

Output:
Return ONLY JSON with keys:
- family_name: 1-5 important tokens that define the base product (e.g., "FELTCRAFT DOLL", "MARSHMALLOWS BOWL SMALL", "CHERRY LIGHTS", "RECORD FRAME").
  * Must include at least one product-type noun if present (DOLL/BOWL/WREATH/LIGHTS/FRAME/MUG/BAG/etc).
  * Must NOT include colors, personal names, odor, flavor.
  * Contain size info if any (e.g., "small", "large", "XL", "S").
- variant_hint: put removed variant info here (colors, personal names, minor descriptors).
No markdown. No extra keys.
"""

def normalize_desc(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = " ".join(s.split())
    return s

STOPWORDS = {"THE","A","AN","AND","&","OF","FOR","WITH","IN","ON","TO"}

def heuristic_fallback(desc: str) -> dict:

    tokens = re.findall(r"[A-Z0-9]+", desc.upper())
    if not tokens:
        return {"family_name": desc, "variant_hint": ""}

    fam = []
    var = []

    for t in tokens:
        if t in STOPWORDS:
            continue
        if t in COLOR_WORDS:
            var.append(t)
            continue
        if t.isalpha() and 3 <= len(t) <= 8 and t not in {"SET","PACK"} and t not in COLOR_WORDS:
            var.append(t)
            continue
        fam.append(t)

    if len(fam) > 4:
        fam = fam[:4]

    family_name = " ".join(fam) if fam else desc
    variant_hint = " ".join(var[:6])
    return {"family_name": family_name, "variant_hint": variant_hint}

def postprocess_family(desc: str, out: dict) -> dict:
    fam = (out.get("family_name") or "").strip()
    var = (out.get("variant_hint") or "").strip()

    if not fam:
        return heuristic_fallback(desc)

    fam_tokens = fam.upper().split()
    cleaned, moved = [], []
    for t in fam_tokens:
        if t in COLOR_WORDS:
            moved.append(t)
        else:
            cleaned.append(t)

    fam = " ".join(cleaned).strip()
    if moved:
        var = (var + " " + " ".join(moved)).strip()

    if not fam:
        return heuristic_fallback(desc)

    return {"family_name": fam, "variant_hint": var}

def load_checkpoint(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            m[obj["desc"]] = obj["result"]
    return m

def append_checkpoint(path: str, items: List[Dict]):
    with open(path, "a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def classify_batch(descs: List[str]) -> Dict[str, dict]:
    """
   return JSON object，template：
    {
      "items": [
        {"desc":"...", "family_name":"...", "variant_hint":"..."},
        ...
      ]
    }
    """
    user_payload = {
        "items": [{"desc": d} for d in descs]
    }

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": (
                "Classify each item. Return ONLY JSON with key 'items' as a list. "
                "Each list element must have keys: desc, family_name, variant_hint.\n"
                f"Input JSON:\n{json.dumps(user_payload, ensure_ascii=False)}"
            )}
        ],
        response_format={"type": "json_object"},
        temperature=TEMPERATURE
    )

    raw = json.loads(resp.choices[0].message.content)
    items = raw.get("items", [])
    out = {}

    got = set()
    for it in items:
        d = normalize_desc(it.get("desc", ""))
        if not d:
            continue
        got.add(d)
        out[d] = postprocess_family(d, it)

    for d in descs:
        if d not in got:
            out[d] = heuristic_fallback(d)

    return out

def collect_unique_descriptions() -> List[str]:
    uniq = set()
    for sh in SHEETS:
        df = pd.read_excel(
            INPUT_XLSX,
            sheet_name=sh,
            engine="openpyxl",
            usecols=["Description"]
        )
        s = df["Description"].map(normalize_desc)
        for d in s.dropna().tolist():
            if d:
                uniq.add(d)
        del df
    uniq_list = sorted(list(uniq))
    return uniq_list

def write_parquet_per_sheet(desc2fam: Dict[str, dict]):
    for sh in SHEETS:
        df = pd.read_excel(INPUT_XLSX, sheet_name=sh, engine="openpyxl")
        df["Description"] = df["Description"].map(normalize_desc)

        df["product_family_name"] = df["Description"].map(
            lambda x: desc2fam.get(x, {}).get("family_name", x)
        )
        df["variant_hint"] = df["Description"].map(
            lambda x: desc2fam.get(x, {}).get("variant_hint", "")
        )

        out_path = os.path.join(OUT_DIR, f"{sh.replace(' ', '_')}.parquet")
        for col in ["Invoice", "StockCode"]:
            if col in df.columns:
                df[col] = df[col].astype("string") 
        if "Customer ID" in df.columns:
            df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce").astype("Int64")
        df.to_parquet(out_path, index=False)
        print(f"Saved parquet: {out_path} | rows={len(df)}")

def main():
    unique_descs = collect_unique_descriptions()
    print(f"Unique descriptions (full dataset): {len(unique_descs)}")

    desc2fam = load_checkpoint(CHECKPOINT_JSONL)
    done = set(desc2fam.keys())
    todo = [d for d in unique_descs if d not in done]
    print(f"Already classified: {len(done)} | Remaining: {len(todo)}")

    pbar = tqdm(total=len(todo), desc="Classifying unique descriptions (batched)")
    i = 0
    while i < len(todo):
        batch = todo[i:i+BATCH_SIZE]
        i += BATCH_SIZE

        try:
            batch_out = classify_batch(batch)
        except Exception:
            batch_out = {d: heuristic_fallback(d) for d in batch}

        ck_items = []
        for d, r in batch_out.items():
            desc2fam[d] = r
            ck_items.append({"desc": d, "result": r})
        append_checkpoint(CHECKPOINT_JSONL, ck_items)

        pbar.update(len(batch))
        time.sleep(SLEEP_SEC)

    pbar.close()
    print(f"Classification complete. Total mapped: {len(desc2fam)}")

    write_parquet_per_sheet(desc2fam)

if __name__ == "__main__":
    main()