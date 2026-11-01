import duckdb

con = duckdb.connect("data/chem.duckdb")

con.execute("""
    CREATE OR REPLACE TABLE raw AS
    SELECT * FROM read_csv_auto('data/for_assessment.csv')
""")

print(con.execute("SELECT COUNT(*) FROM raw").fetchone())

print(con.execute("DESCRIBE raw").fetchdf())

con.execute("""
    CREATE OR REPLACE TABLE clean AS
    SELECT DISTINCT * FROM raw
""")

print("Raw rows:  ", con.execute("SELECT COUNT(*) FROM raw").fetchone()[0])
print("Clean rows:", con.execute("SELECT COUNT(*) FROM clean").fetchone()[0])

print(con.execute("""
    SELECT CasNumber, COUNT(*) AS n
    FROM clean
    WHERE ChemicalName = 'Titanium dioxide'
    GROUP BY CasNumber
    ORDER BY n DESC
""").fetchdf())