#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Mapping from .dat filename to (table_name, columns)
DAT_TABLES: Dict[str, Tuple[str, List[str]]] = {
    "3055.dat": ("customer", [
        "customer_id","store_id","first_name","last_name","email",
        "address_id","activebool","create_date","last_update","active"
    ]),
    "3057.dat": ("actor", ["actor_id","first_name","last_name","last_update"]),
    "3059.dat": ("category", ["category_id","name","last_update"]),
    "3061.dat": ("film", [
        "film_id","title","description","release_year","language_id",
        "rental_duration","rental_rate","length","replacement_cost",
        "rating","last_update","special_features","fulltext"
    ]),
    "3062.dat": ("film_actor", ["actor_id","film_id","last_update"]),
    "3063.dat": ("film_category", ["film_id","category_id","last_update"]),
    "3065.dat": ("address", [
        "address_id","address","address2","district","city_id","postal_code","phone","last_update"
    ]),
    "3067.dat": ("city", ["city_id","city","country_id","last_update"]),
    "3069.dat": ("country", ["country_id","country","last_update"]),
    "3071.dat": ("inventory", ["inventory_id","film_id","store_id","last_update"]),
    "3073.dat": ("language", ["language_id","name","last_update"]),
    "3075.dat": ("payment", ["payment_id","customer_id","staff_id","rental_id","amount","payment_date"]),
    "3077.dat": ("rental", ["rental_id","rental_date","inventory_id","customer_id","return_date","staff_id","last_update"]),
    "3079.dat": ("staff", ["staff_id","first_name","last_name","address_id","email","store_id","active","username","password","last_update","picture"]),
    "3081.dat": ("store", ["store_id","manager_staff_id","address_id","last_update"]),
}


def load_dat_file(path: Path, columns: List[str], nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=columns,
        na_values="\\N",
        engine="python",
        nrows=nrows,
    )


def build_html_preview(table_name: str, df: pd.DataFrame, max_rows: int = 5) -> str:
    parts: List[str] = []
    parts.append(f"<h2>{table_name}</h2>")
    parts.append(f"<p>Shape: {df.shape[0]} rows × {df.shape[1]} cols (showing first {min(max_rows, len(df))})</p>")
    parts.append(df.head(max_rows).to_html(index=False))

    # Basic stats for numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        desc = df[num_cols].describe().T
        parts.append("<h4>Numeric columns (describe)</h4>")
        parts.append(desc.to_html())

    # A few top value counts for first categorical column
    cat_cols = [c for c in df.columns if c not in num_cols]
    if cat_cols:
        first_cat = cat_cols[0]
        vc = df[first_cat].value_counts(dropna=False).head(10)
        parts.append(f"<h4>Top values for '{first_cat}'</h4>")
        parts.append(vc.to_frame(name="count").to_html())

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Preview dvdrental .dat files with basic info and HTML report")
    parser.add_argument("--root", default=".", help="Folder containing .dat files (default: current directory)")
    parser.add_argument("--out", default="dat_preview.html", help="Output HTML report path")
    parser.add_argument("--rows", type=int, default=5000, help="Max rows to read per file (for speed)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    html_sections: List[str] = []

    print(f"Scanning {root} for .dat files...\n")
    for fname, (table, columns) in DAT_TABLES.items():
        fpath = root / fname
        if not fpath.exists():
            continue
        print(f"Loading {fname} -> table '{table}' ...")
        df = load_dat_file(fpath, columns, nrows=args.rows)
        print(f"  - shape: {df.shape}")
        print(f"  - head:\n{df.head(5)}\n")
        html_sections.append(build_html_preview(table, df))

    if not html_sections:
        print("No known .dat files found in the specified folder.")
        return

    html_doc = (
        "<html><head><meta charset='utf-8'><title>dvdrental .dat preview</title>"
        "<style>body{font-family:Arial, sans-serif; margin:20px;} table{border-collapse:collapse;}"
        "table, th, td{border:1px solid #ddd; padding:4px;} th{background:#f5f5f5;}</style>"
        "</head><body>"
        "<h1>dvdrental .dat preview</h1>"
        + "\n".join(html_sections)
        + "</body></html>"
    )

    out_path = Path(args.out).resolve()
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"\nHTML report saved to: {out_path}")


if __name__ == "__main__":
    main() 