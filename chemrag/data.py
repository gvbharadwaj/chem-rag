"""Load the CSV into DuckDB and clean it (dedupe, fix CAS, backfill CAS, flag date conflicts)."""

import os
import re

import duckdb

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "for_assessment.csv")
DB_PATH = os.path.join(DATA_DIR, "chemrag_app.duckdb")


def normalize_cas(raw):
    """'CAS #13463-67-7 ' -> '13463-67-7'. Returns None if invalid (check digit)."""
    if raw is None:
        return None
    s = re.sub(r"(?i)cas|rn|[#:\s]", "", str(raw))
    if re.fullmatch(r"\d{5,10}", s):  # digits only
        s = f"{s[:-3]}-{s[-3:-1]}-{s[-1]}"
    if not re.fullmatch(r"\d{2,7}-\d{2}-\d", s):
        return None
    digits = [int(d) for d in s.replace("-", "")]
    check = sum(d * (i + 1) for i, d in enumerate(reversed(digits[:-1]))) % 10
    return s if check == digits[-1] else None


def build_database():
    con = duckdb.connect(DB_PATH)
    con.execute(f"CREATE OR REPLACE TABLE raw AS SELECT * FROM read_csv_auto('{CSV_PATH}')")
    con.execute("CREATE OR REPLACE TABLE dedup AS SELECT DISTINCT * FROM raw")

    # clean CAS lookup (only ~125 distinct values)
    con.execute("CREATE OR REPLACE TABLE cas_map (raw_cas VARCHAR, cas VARCHAR)")
    raw_cas = con.execute("SELECT DISTINCT CasNumber FROM dedup WHERE CasNumber IS NOT NULL").fetchall()
    con.executemany("INSERT INTO cas_map VALUES (?, ?)", [(r[0], normalize_cas(r[0])) for r in raw_cas])

    # most common clean CAS per chemical name -> used to fill missing CAS
    con.execute("""
        CREATE OR REPLACE TABLE name_cas AS
        SELECT TRIM(d.ChemicalName) AS ChemicalName, MODE(m.cas) AS cas
        FROM dedup d JOIN cas_map m ON d.CasNumber = m.raw_cas
        WHERE m.cas IS NOT NULL GROUP BY 1
    """)
    con.execute("""
        CREATE OR REPLACE TABLE clean AS
        SELECT d.*, COALESCE(m.cas, n.cas) AS cas_clean,
               COALESCE(d.DiscontinuedDate < d.InitialDateReported, FALSE) AS date_conflict
        FROM dedup d
        LEFT JOIN cas_map m ON d.CasNumber = m.raw_cas
        LEFT JOIN name_cas n ON TRIM(d.ChemicalName) = n.ChemicalName
    """)
    con.close()


def get_connection(rebuild=False):
    """Open the database; (re)build it if missing or if it doesn't have our tables."""
    if not rebuild and os.path.exists(DB_PATH):
        con = duckdb.connect(DB_PATH)
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        con.close()
        rebuild = not {"clean", "name_cas", "raw"} <= tables
    if rebuild or not os.path.exists(DB_PATH):
        build_database()
    return duckdb.connect(DB_PATH, read_only=True)
