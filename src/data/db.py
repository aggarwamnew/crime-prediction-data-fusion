"""
DuckDB database manager for the thesis project.

Provides a single connection and table management for:
- Crime data (loaded from CSVs)
- LSOA boundaries (loaded from shapefiles/GeoJSON)
- IMD data (loaded from CSV)
- Fused analysis tables

Usage:
    from src.data.db import ThesisDB

    db = ThesisDB()
    db.ingest_crime_csvs("data/raw/london/crime/")
    df = db.query("SELECT * FROM crime LIMIT 10")
"""

import duckdb
import pandas as pd
from pathlib import Path
from src.utils.config import DB_PATH


class ThesisDB:
    """Thin wrapper around DuckDB for the thesis project."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        # Install and load spatial extension for later use
        self._setup_extensions()

    def _setup_extensions(self):
        """Install DuckDB extensions we'll need."""
        try:
            self.conn.execute("INSTALL spatial; LOAD spatial;")
        except Exception:
            # Extension might already be installed
            try:
                self.conn.execute("LOAD spatial;")
            except Exception:
                print("⚠️  DuckDB spatial extension not available. "
                      "Spatial queries will not work.")

    def query(self, sql: str) -> pd.DataFrame:
        """Run a SQL query and return a pandas DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def execute(self, sql: str):
        """Run a SQL statement (no return value)."""
        self.conn.execute(sql)

    def ingest_crime_csvs(self, csv_dir: str | Path, table_name: str = "crime"):
        """
        Ingest all crime CSV files from a directory into a DuckDB table.

        The data.police.uk CSVs have this schema:
            Crime ID, Month, Reported by, Falls within, Longitude, Latitude,
            Location, LSOA code, LSOA name, Crime type, Last outcome category,
            Context

        Args:
            csv_dir: Path to directory containing CSV files.
            table_name: Name of the DuckDB table to create.
        """
        csv_dir = Path(csv_dir)
        csv_files = sorted(csv_dir.glob("**/*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")

        print(f"📂 Found {len(csv_files)} CSV files in {csv_dir}")

        # DuckDB can read multiple CSVs with a glob pattern
        glob_pattern = str(csv_dir / "**" / "*.csv")

        self.conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT
                "Crime ID"              AS crime_id,
                "Month"                 AS month,
                "Reported by"           AS reported_by,
                "Falls within"          AS falls_within,
                CAST("Longitude" AS DOUBLE) AS longitude,
                CAST("Latitude" AS DOUBLE)  AS latitude,
                "Location"              AS location,
                "LSOA code"             AS lsoa_code,
                "LSOA name"             AS lsoa_name,
                "Crime type"            AS crime_type,
                "Last outcome category" AS outcome,
                "Context"               AS context
            FROM read_csv('{glob_pattern}',
                         header=true,
                         ignore_errors=true,
                         auto_detect=true)
        """)

        count = self.conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        print(f"✅ Loaded {count:,} rows into '{table_name}' table")

        # Print summary
        summary = self.query(f"""
            SELECT
                MIN(month) AS earliest_month,
                MAX(month) AS latest_month,
                COUNT(DISTINCT month) AS n_months,
                COUNT(DISTINCT lsoa_code) AS n_lsoas,
                COUNT(DISTINCT crime_type) AS n_crime_types,
                COUNT(*) AS total_crimes
            FROM {table_name}
        """)
        print(f"\n📊 Summary:")
        print(f"   Date range:  {summary['earliest_month'].iloc[0]} → "
              f"{summary['latest_month'].iloc[0]}")
        print(f"   Months:      {summary['n_months'].iloc[0]}")
        print(f"   LSOAs:       {summary['n_lsoas'].iloc[0]:,}")
        print(f"   Crime types: {summary['n_crime_types'].iloc[0]}")
        print(f"   Total rows:  {summary['total_crimes'].iloc[0]:,}")

        return count

    def table_info(self, table_name: str = "crime") -> pd.DataFrame:
        """Get column info for a table."""
        return self.query(f"DESCRIBE {table_name}")

    def tables(self) -> list[str]:
        """List all tables in the database."""
        result = self.query("SHOW TABLES")
        return result["name"].tolist()

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        tables = self.tables()
        return (f"ThesisDB(path='{self.db_path}', "
                f"tables={tables})")
