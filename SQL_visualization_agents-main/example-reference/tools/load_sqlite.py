#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sqlalchemy import create_engine


def load_dat(path: Path, columns: List[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=columns, na_values="\\N", engine="python")


def main():
    root = Path(os.environ.get("DVDRENTAL_DIR", ".")).resolve()
    sqlite_path = root / "dvdrental.sqlite"
    engine = create_engine(f"sqlite:///{sqlite_path}")

    # Minimal set of tables to enable common queries; extend as needed
    # Define columns in the order of the .dat files per restore.sql mapping
    print(f"Loading .dat files from: {root}")

    # actor: 3057.dat
    actor_cols = ["actor_id","first_name","last_name","last_update"]
    actor = load_dat(root / "3057.dat", actor_cols)
    actor.to_sql("actor", engine, index=False, if_exists="replace")

    # category: 3059.dat
    category_cols = ["category_id","name","last_update"]
    category = load_dat(root / "3059.dat", category_cols)
    category.to_sql("category", engine, index=False, if_exists="replace")

    # film: 3061.dat
    film_cols = [
        "film_id","title","description","release_year","language_id",
        "rental_duration","rental_rate","length","replacement_cost",
        "rating","last_update","special_features","fulltext"
    ]
    film = load_dat(root / "3061.dat", film_cols)
    film.to_sql("film", engine, index=False, if_exists="replace")

    # film_actor: 3062.dat
    film_actor_cols = ["actor_id","film_id","last_update"]
    film_actor = load_dat(root / "3062.dat", film_actor_cols)
    film_actor.to_sql("film_actor", engine, index=False, if_exists="replace")

    # film_category: 3063.dat
    film_category_cols = ["film_id","category_id","last_update"]
    film_category = load_dat(root / "3063.dat", film_category_cols)
    film_category.to_sql("film_category", engine, index=False, if_exists="replace")

    # address: 3065.dat
    address_cols = ["address_id","address","address2","district","city_id","postal_code","phone","last_update"]
    address = load_dat(root / "3065.dat", address_cols)
    address.to_sql("address", engine, index=False, if_exists="replace")

    # city: 3067.dat
    city_cols = ["city_id","city","country_id","last_update"]
    city = load_dat(root / "3067.dat", city_cols)
    city.to_sql("city", engine, index=False, if_exists="replace")

    # country: 3069.dat
    country_cols = ["country_id","country","last_update"]
    country = load_dat(root / "3069.dat", country_cols)
    country.to_sql("country", engine, index=False, if_exists="replace")

    # inventory: 3071.dat
    inventory_cols = ["inventory_id","film_id","store_id","last_update"]
    inventory = load_dat(root / "3071.dat", inventory_cols)
    inventory.to_sql("inventory", engine, index=False, if_exists="replace")

    # language: 3073.dat
    language_cols = ["language_id","name","last_update"]
    language = load_dat(root / "3073.dat", language_cols)
    language.to_sql("language", engine, index=False, if_exists="replace")

    # customer: 3055.dat (was missing!)
    customer_cols = ["customer_id","store_id","first_name","last_name","email","address_id","activebool","create_date","last_update","active"]
    customer = load_dat(root / "3055.dat", customer_cols)
    customer.to_sql("customer", engine, index=False, if_exists="replace")

    # payment: 3075.dat
    payment_cols = ["payment_id","customer_id","staff_id","rental_id","amount","payment_date"]
    payment = load_dat(root / "3075.dat", payment_cols)
    payment.to_sql("payment", engine, index=False, if_exists="replace")

    # rental: 3077.dat
    rental_cols = ["rental_id","rental_date","inventory_id","customer_id","return_date","staff_id","last_update"]
    rental = load_dat(root / "3077.dat", rental_cols)
    rental.to_sql("rental", engine, index=False, if_exists="replace")

    # staff: 3079.dat
    staff_cols = ["staff_id","first_name","last_name","address_id","email","store_id","active","username","password","last_update","picture"]
    staff = load_dat(root / "3079.dat", staff_cols)
    staff.to_sql("staff", engine, index=False, if_exists="replace")

    # store: 3081.dat
    store_cols = ["store_id","manager_staff_id","address_id","last_update"]
    store = load_dat(root / "3081.dat", store_cols)
    store.to_sql("store", engine, index=False, if_exists="replace")

    print(f"SQLite database created at: {sqlite_path}")


if __name__ == "__main__":
    main() 