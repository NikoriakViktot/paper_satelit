"""
Neo4j writer for the flood mapping extraction pipeline.

Converts ExtractionResult objects into graph nodes and relationships
using the schema defined in neo4j_schema.cypher.

Usage
-----
    from src.graphstore.neo4j_writer import Neo4jWriter

    with Neo4jWriter() as gdb:
        gdb.seed_vocabulary()          # run once on a fresh database
        gdb.write_paper(result)        # write one ExtractionResult
        gdb.write_papers(results)      # write a list
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.extraction.base import ExtractionResult

logger = logging.getLogger(__name__)

_NEO4J_URI  = os.getenv("NEO4J_URI",     "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER",    "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "python2024")


class Neo4jWriter:
    """
    Thin wrapper around the Neo4j Python driver.

    Requires the `neo4j` package:
        pip install neo4j
    """

    def __init__(
        self,
        uri: str = _NEO4J_URI,
        user: str = _NEO4J_USER,
        password: str = _NEO4J_PASS,
    ) -> None:
        from neo4j import GraphDatabase  # type: ignore[import]
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Neo4j driver connected to %s", uri)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def seed_vocabulary(self) -> None:
        """
        Create shared lookup nodes: SensorType, TaskType, Metric, Method,
        Satellite, DataProduct.  Safe to run repeatedly (MERGE is idempotent).
        """
        with self._driver.session() as s:
            s.execute_write(_seed_sensor_types)
            s.execute_write(_seed_task_types)
            s.execute_write(_seed_metrics)
            s.execute_write(_seed_methods)
            s.execute_write(_seed_satellites)
            s.execute_write(_seed_data_products)
        logger.info("Vocabulary seeded.")

    def write_paper(self, result: "ExtractionResult") -> None:
        with self._driver.session() as s:
            s.execute_write(_upsert_paper, result)
        logger.debug("Written: %s", result.source_file)

    def write_papers(self, results: list["ExtractionResult"]) -> None:
        ok = err = 0
        for r in results:
            try:
                self.write_paper(r)
                ok += 1
            except Exception as exc:
                logger.warning("Failed to write %s: %s", r.source_file, exc)
                err += 1
        logger.info("Graph write: %d OK, %d errors", ok, err)

    def create_constraints(self) -> None:
        """Create uniqueness constraints. Run once on a fresh database."""
        constraints = [
            "CREATE CONSTRAINT paper_source_file IF NOT EXISTS FOR (p:Paper) REQUIRE p.source_file IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT doi_value IF NOT EXISTS FOR (d:DOI) REQUIRE d.value IS UNIQUE",
            "CREATE CONSTRAINT satellite_name IF NOT EXISTS FOR (s:Satellite) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT sensor_type IF NOT EXISTS FOR (st:SensorType) REQUIRE st.type IS UNIQUE",
            "CREATE CONSTRAINT data_product_code IF NOT EXISTS FOR (dp:DataProduct) REQUIRE dp.code IS UNIQUE",
            "CREATE CONSTRAINT method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT task_type_name IF NOT EXISTS FOR (t:TaskType) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT study_area_country IF NOT EXISTS FOR (sa:StudyArea) REQUIRE sa.country IS UNIQUE",
            "CREATE CONSTRAINT event_name IF NOT EXISTS FOR (e:Event) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT metric_type IF NOT EXISTS FOR (mt:Metric) REQUIRE mt.type IS UNIQUE",
        ]
        with self._driver.session() as s:
            for cypher in constraints:
                s.run(cypher)
        logger.info("Constraints created.")


# ── Transaction functions ─────────────────────────────────────────────────────

def _upsert_paper(tx, result: "ExtractionResult") -> None:
    # 1. Paper node
    tx.run("""
        MERGE (p:Paper {source_file: $source_file})
        SET p.title          = $title,
            p.year           = $year,
            p.abstract       = $abstract,
            p.near_real_time = $near_real_time,
            p.latency        = $latency,
            p.revisit_time   = $revisit_time,
            p.confidence     = $confidence
    """, source_file=result.source_file,
         title=result.title or None,
         year=int(result.year) if result.year and result.year.isdigit() else None,
         abstract=result.abstract or None,
         near_real_time=result.near_real_time,
         latency=result.latency or None,
         revisit_time=result.revisit_time or None,
         confidence=result.confidence)

    # 2. Authors
    if result.authors:
        for name in _split_csv(result.authors):
            tx.run("""
                MERGE (a:Author {name: $name})
                WITH a
                MATCH (p:Paper {source_file: $sf})
                MERGE (p)-[:AUTHORED_BY]->(a)
            """, name=name, sf=result.source_file)

    # 3. DOI
    if result.doi:
        tx.run("""
            MERGE (d:DOI {value: $doi})
            WITH d
            MATCH (p:Paper {source_file: $sf})
            MERGE (p)-[:HAS_DOI]->(d)
        """, doi=result.doi, sf=result.source_file)

    # 4. Task type
    if result.study_type:
        tx.run("""
            MERGE (tt:TaskType {name: $name})
            WITH tt
            MATCH (p:Paper {source_file: $sf})
            MERGE (p)-[:HAS_TASK_TYPE]->(tt)
        """, name=result.study_type, sf=result.source_file)

    # 5. Satellites
    if result.satellite_names:
        for sat_name in _split_csv(result.satellite_names):
            tx.run("""
                MERGE (s:Satellite {name: $name})
                WITH s
                MATCH (p:Paper {source_file: $sf})
                MERGE (p)-[:USES_SATELLITE]->(s)
            """, name=sat_name, sf=result.source_file)

    # 6. Sensor type (linked to Paper directly for fast filtering)
    if result.sensor_type:
        tx.run("""
            MERGE (st:SensorType {type: $stype})
            WITH st
            MATCH (p:Paper {source_file: $sf})
            MERGE (p)-[:HAS_SENSOR_TYPE]->(st)
        """, stype=result.sensor_type, sf=result.source_file)

    # 7. Data products
    if result.data_product:
        for code in _split_csv(result.data_product):
            tx.run("""
                MERGE (dp:DataProduct {code: $code})
                WITH dp
                MATCH (p:Paper {source_file: $sf})
                MERGE (p)-[:USES_DATA_PRODUCT]->(dp)
            """, code=code, sf=result.source_file)

    # 8. Methods
    if result.methods:
        for method_name in _split_csv(result.methods):
            tx.run("""
                MERGE (m:Method {name: $name})
                WITH m
                MATCH (p:Paper {source_file: $sf})
                MERGE (p)-[:USES_METHOD]->(m)
            """, name=method_name, sf=result.source_file)

    # 9. Study area
    if result.country:
        tx.run("""
            MERGE (sa:StudyArea {country: $country})
            SET sa.region      = COALESCE($region, sa.region),
                sa.river_basin = COALESCE($river_basin, sa.river_basin)
            WITH sa
            MATCH (p:Paper {source_file: $sf})
            MERGE (p)-[:STUDIES_AREA]->(sa)
        """, country=result.country,
             region=result.region or None,
             river_basin=result.river_basin or None,
             sf=result.source_file)

    # 10. Event
    if result.city_event:
        year = int(result.year) if result.year and result.year.isdigit() else None
        tx.run("""
            MERGE (ev:Event {name: $name})
            SET ev.year    = COALESCE($year, ev.year),
                ev.country = COALESCE($country, ev.country)
            WITH ev
            MATCH (p:Paper {source_file: $sf})
            MERGE (p)-[:ANALYZES_EVENT]->(ev)
        """, name=result.city_event,
             year=year,
             country=result.country or None,
             sf=result.source_file)

    # 11. Metrics (optional — only write when value is present)
    _metrics = {
        "OA":    result.oa,
        "F1":    result.f1,
        "IoU":   result.iou,
        "Kappa": result.kappa,
    }
    for mtype, mvalue in _metrics.items():
        if mvalue is not None:
            tx.run("""
                MATCH (p:Paper {source_file: $sf})
                MERGE (mt:Metric {type: $mtype})
                MERGE (p)-[r:REPORTS_METRIC]->(mt)
                SET r.value = $value
            """, sf=result.source_file, mtype=mtype, value=mvalue)


# ── Vocabulary seeders ────────────────────────────────────────────────────────

def _seed_sensor_types(tx) -> None:
    for t in ("SAR", "Optical", "Multi-sensor"):
        tx.run("MERGE (:SensorType {type: $t})", t=t)


def _seed_task_types(tx) -> None:
    for name in (
        "Satellite flood mapping", "ML/DL classification",
        "Hydrological forecasting", "Hydraulic modeling",
        "Operational mapping system", "Review paper", "Dataset/benchmark paper",
    ):
        tx.run("MERGE (:TaskType {name: $name})", name=name)


def _seed_metrics(tx) -> None:
    for mtype, desc in (
        ("OA",    "Overall Accuracy"),
        ("F1",    "F1 Score"),
        ("IoU",   "Intersection over Union"),
        ("Kappa", "Cohen's Kappa"),
    ):
        tx.run("MERGE (m:Metric {type: $t}) SET m.description = $d", t=mtype, d=desc)


def _seed_methods(tx) -> None:
    methods = [
        ("Thresholding",        "SAR-processing"),
        ("Change detection",    "SAR-processing"),
        ("NDWI/MNDWI",          "Index"),
        ("Random Forest",       "ML"),
        ("SVM",                 "ML"),
        ("Maximum likelihood",  "ML"),
        ("U-Net",               "DL"),
        ("CNN",                 "DL"),
        ("LSTM",                "DL"),
        ("Transformer",         "DL"),
        ("OBIA",                "SAR-processing"),
        ("Hydrodynamic model",  "Hydrodynamic"),
        ("Operational workflow","Operational"),
    ]
    for name, cat in methods:
        tx.run("MERGE (m:Method {name: $name}) SET m.category = $cat", name=name, cat=cat)


def _seed_satellites(tx) -> None:
    sar_sats = [
        "Sentinel-1", "TerraSAR-X", "COSMO-SkyMed",
        "ALOS-2", "RADARSAT-2", "UAVSAR", "ICEYE",
    ]
    opt_sats = [
        "Sentinel-2", "Landsat-8", "Landsat-9",
        "MODIS", "VIIRS", "WorldView-2", "WorldView-3", "Pleiades",
    ]
    for name in sar_sats:
        tx.run("""
            MERGE (s:Satellite {name: $name})
            MERGE (st:SensorType {type: 'SAR'})
            MERGE (s)-[:HAS_SENSOR_TYPE]->(st)
        """, name=name)
    for name in opt_sats:
        tx.run("""
            MERGE (s:Satellite {name: $name})
            MERGE (st:SensorType {type: 'Optical'})
            MERGE (s)-[:HAS_SENSOR_TYPE]->(st)
        """, name=name)


def _seed_data_products(tx) -> None:
    products = [
        ("GRD",  "Sentinel-1"),
        ("SLC",  "Sentinel-1"),
        ("MSI",  "Sentinel-2"),
        ("OLI",  "Landsat-8"),
        ("TIRS", "Landsat-8"),
    ]
    for code, sat in products:
        tx.run("""
            MERGE (dp:DataProduct {code: $code})
            MERGE (s:Satellite {name: $sat})
            MERGE (s)-[:PROVIDES_PRODUCT]->(dp)
        """, code=code, sat=sat)


# ── Utility ───────────────────────────────────────────────────────────────────

def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]
