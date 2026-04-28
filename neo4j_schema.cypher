// ═══════════════════════════════════════════════════════════════════════════════
// Neo4j Graph Schema
// Satellite-Based Flood Mapping — Scientific Literature Review
//
// Connection (Python driver):
//   bolt://localhost:7687
//   auth: ("neo4j", "python2024")
//
// Browser:  http://localhost:7474
// ═══════════════════════════════════════════════════════════════════════════════


// ───────────────────────────────────────────────────────────────────────────────
// SECTION 1 — NODE LABELS AND PROPERTIES (schema documentation)
// ───────────────────────────────────────────────────────────────────────────────
//
// (:Paper)
//   source_file      String    — original PDF filename  [UNIQUE]
//   title            String
//   year             Integer
//   abstract         String
//   near_real_time   Boolean   — true | false | null
//   latency          String    — e.g. "6 h", "1–3 days"
//   revisit_time     String    — e.g. "6 days"
//   confidence       Float     — extraction quality 0–1
//
// (:Author)
//   name             String    [UNIQUE]
//
// (:DOI)
//   value            String    [UNIQUE]   — e.g. "10.1016/j.rse.2022.112345"
//
// (:Satellite)
//   name             String    [UNIQUE]   — e.g. "Sentinel-1"
//
// (:SensorType)
//   type             String    [UNIQUE]   — SAR | Optical | Multi-sensor
//
// (:DataProduct)
//   code             String    [UNIQUE]   — e.g. "GRD", "MSI", "OLI"
//
// (:Method)
//   name             String    [UNIQUE]   — e.g. "U-Net", "Thresholding"
//   category         String    — DL | ML | Index | SAR-processing | Hydrodynamic | Operational
//
// (:TaskType)
//   name             String    [UNIQUE]
//   — Satellite flood mapping | ML/DL classification | Hydrological forecasting
//     Hydraulic modeling | Operational mapping system | Review paper
//     Dataset/benchmark paper
//
// (:StudyArea)
//   country          String    [UNIQUE key for MERGE]
//   region           String    — sub-national area or continent
//   river_basin      String
//
// (:Event)
//   name             String    [UNIQUE]   — e.g. "2021 Germany floods"
//   year             Integer
//   country          String
//
// (:Metric)
//   type             String    [UNIQUE]   — OA | F1 | IoU | Kappa
//   description      String    — human-readable name
//
//   NOTE: the numeric VALUE lives on the relationship, not on this node.
//   (:Paper)-[:REPORTS_METRIC {value: 0.92}]->(:Metric {type: "OA"})
//   This lets Metric nodes be shared across all papers.


// ───────────────────────────────────────────────────────────────────────────────
// SECTION 2 — RELATIONSHIP TYPES
// ───────────────────────────────────────────────────────────────────────────────
//
// (:Paper)-[:AUTHORED_BY]->(:Author)
// (:Paper)-[:HAS_DOI]->(:DOI)
// (:Paper)-[:HAS_TASK_TYPE]->(:TaskType)
// (:Paper)-[:USES_SATELLITE]->(:Satellite)
// (:Paper)-[:USES_DATA_PRODUCT]->(:DataProduct)
// (:Paper)-[:USES_METHOD]->(:Method)
// (:Paper)-[:STUDIES_AREA]->(:StudyArea)
// (:Paper)-[:ANALYZES_EVENT]->(:Event)
// (:Paper)-[:REPORTS_METRIC {value: Float}]->(:Metric)   ← OPTIONAL
//
// (:Satellite)-[:HAS_SENSOR_TYPE]->(:SensorType)
// (:Satellite)-[:PROVIDES_PRODUCT]->(:DataProduct)


// ───────────────────────────────────────────────────────────────────────────────
// SECTION 3 — UNIQUENESS CONSTRAINTS
// Run once on a fresh database.
// ───────────────────────────────────────────────────────────────────────────────

CREATE CONSTRAINT paper_source_file IF NOT EXISTS
  FOR (p:Paper) REQUIRE p.source_file IS UNIQUE;

CREATE CONSTRAINT author_name IF NOT EXISTS
  FOR (a:Author) REQUIRE a.name IS UNIQUE;

CREATE CONSTRAINT doi_value IF NOT EXISTS
  FOR (d:DOI) REQUIRE d.value IS UNIQUE;

CREATE CONSTRAINT satellite_name IF NOT EXISTS
  FOR (s:Satellite) REQUIRE s.name IS UNIQUE;

CREATE CONSTRAINT sensor_type IF NOT EXISTS
  FOR (st:SensorType) REQUIRE st.type IS UNIQUE;

CREATE CONSTRAINT data_product_code IF NOT EXISTS
  FOR (dp:DataProduct) REQUIRE dp.code IS UNIQUE;

CREATE CONSTRAINT method_name IF NOT EXISTS
  FOR (m:Method) REQUIRE m.name IS UNIQUE;

CREATE CONSTRAINT task_type_name IF NOT EXISTS
  FOR (t:TaskType) REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT study_area_country IF NOT EXISTS
  FOR (sa:StudyArea) REQUIRE sa.country IS UNIQUE;

CREATE CONSTRAINT event_name IF NOT EXISTS
  FOR (e:Event) REQUIRE e.name IS UNIQUE;

CREATE CONSTRAINT metric_type IF NOT EXISTS
  FOR (mt:Metric) REQUIRE mt.type IS UNIQUE;


// ───────────────────────────────────────────────────────────────────────────────
// SECTION 4 — SEED LOOKUP NODES (run once)
// These are shared vocabulary nodes that all papers reference.
// ───────────────────────────────────────────────────────────────────────────────

// SensorType vocabulary
MERGE (:SensorType {type: "SAR"});
MERGE (:SensorType {type: "Optical"});
MERGE (:SensorType {type: "Multi-sensor"});

// TaskType vocabulary
MERGE (:TaskType {name: "Satellite flood mapping"});
MERGE (:TaskType {name: "ML/DL classification"});
MERGE (:TaskType {name: "Hydrological forecasting"});
MERGE (:TaskType {name: "Hydraulic modeling"});
MERGE (:TaskType {name: "Operational mapping system"});
MERGE (:TaskType {name: "Review paper"});
MERGE (:TaskType {name: "Dataset/benchmark paper"});

// Metric vocabulary (value goes on the relationship)
MERGE (:Metric {type: "OA",    description: "Overall Accuracy"});
MERGE (:Metric {type: "F1",    description: "F1 Score"});
MERGE (:Metric {type: "IoU",   description: "Intersection over Union"});
MERGE (:Metric {type: "Kappa", description: "Cohen's Kappa"});

// Method vocabulary with categories
MERGE (m:Method {name: "Thresholding"})   SET m.category = "SAR-processing";
MERGE (m:Method {name: "Change detection"}) SET m.category = "SAR-processing";
MERGE (m:Method {name: "NDWI/MNDWI"})    SET m.category = "Index";
MERGE (m:Method {name: "Random Forest"}) SET m.category = "ML";
MERGE (m:Method {name: "SVM"})           SET m.category = "ML";
MERGE (m:Method {name: "Maximum likelihood"}) SET m.category = "ML";
MERGE (m:Method {name: "U-Net"})         SET m.category = "DL";
MERGE (m:Method {name: "CNN"})           SET m.category = "DL";
MERGE (m:Method {name: "LSTM"})          SET m.category = "DL";
MERGE (m:Method {name: "Transformer"})   SET m.category = "DL";
MERGE (m:Method {name: "OBIA"})          SET m.category = "SAR-processing";
MERGE (m:Method {name: "Hydrodynamic model"}) SET m.category = "Hydrodynamic";
MERGE (m:Method {name: "Operational workflow"}) SET m.category = "Operational";

// Satellite nodes linked to SensorType
MERGE (s:Satellite {name: "Sentinel-1"})
  WITH s MERGE (st:SensorType {type: "SAR"})
  MERGE (s)-[:HAS_SENSOR_TYPE]->(st);

MERGE (s:Satellite {name: "TerraSAR-X"})
  WITH s MERGE (st:SensorType {type: "SAR"})
  MERGE (s)-[:HAS_SENSOR_TYPE]->(st);

MERGE (s:Satellite {name: "COSMO-SkyMed"})
  WITH s MERGE (st:SensorType {type: "SAR"})
  MERGE (s)-[:HAS_SENSOR_TYPE]->(st);

MERGE (s:Satellite {name: "ALOS-2"})
  WITH s MERGE (st:SensorType {type: "SAR"})
  MERGE (s)-[:HAS_SENSOR_TYPE]->(st);

MERGE (s:Satellite {name: "RADARSAT-2"})
  WITH s MERGE (st:SensorType {type: "SAR"})
  MERGE (s)-[:HAS_SENSOR_TYPE]->(st);

MERGE (s:Satellite {name: "UAVSAR"})
  WITH s MERGE (st:SensorType {type: "SAR"})
  MERGE (s)-[:HAS_SENSOR_TYPE]->(st);

MERGE (s:Satellite {name: "Sentinel-2"})
  WITH s MERGE (st:SensorType {type: "Optical"})
  MERGE (s)-[:HAS_SENSOR_TYPE]->(st);

MERGE (s:Satellite {name: "Landsat-8"})
  WITH s MERGE (st:SensorType {type: "Optical"})
  MERGE (s)-[:HAS_SENSOR_TYPE]->(st);

MERGE (s:Satellite {name: "Landsat-9"})
  WITH s MERGE (st:SensorType {type: "Optical"})
  MERGE (s)-[:HAS_SENSOR_TYPE]->(st);

MERGE (s:Satellite {name: "MODIS"})
  WITH s MERGE (st:SensorType {type: "Optical"})
  MERGE (s)-[:HAS_SENSOR_TYPE]->(st);

MERGE (s:Satellite {name: "VIIRS"})
  WITH s MERGE (st:SensorType {type: "Optical"})
  MERGE (s)-[:HAS_SENSOR_TYPE]->(st);

// DataProduct nodes linked to Satellites
MERGE (dp:DataProduct {code: "GRD"})
  WITH dp MATCH (s:Satellite {name: "Sentinel-1"})
  MERGE (s)-[:PROVIDES_PRODUCT]->(dp);

MERGE (dp:DataProduct {code: "SLC"})
  WITH dp MATCH (s:Satellite {name: "Sentinel-1"})
  MERGE (s)-[:PROVIDES_PRODUCT]->(dp);

MERGE (dp:DataProduct {code: "MSI"})
  WITH dp MATCH (s:Satellite {name: "Sentinel-2"})
  MERGE (s)-[:PROVIDES_PRODUCT]->(dp);

MERGE (dp:DataProduct {code: "OLI"})
  WITH dp MATCH (s:Satellite {name: "Landsat-8"})
  MERGE (s)-[:PROVIDES_PRODUCT]->(dp);

MERGE (dp:DataProduct {code: "TIRS"})
  WITH dp MATCH (s:Satellite {name: "Landsat-8"})
  MERGE (s)-[:PROVIDES_PRODUCT]->(dp);


// ───────────────────────────────────────────────────────────────────────────────
// SECTION 5 — EXAMPLE: INSERT ONE PAPER
// Based on: Smith & Jones (2022), Bangladesh / Sentinel-1 / SAR thresholding
// ───────────────────────────────────────────────────────────────────────────────

// 5a. Paper node
MERGE (p:Paper {source_file: "Smith_2022_Bangladesh_SAR.pdf"})
SET
  p.title          = "Near-Real-Time Flood Mapping with Sentinel-1 SAR over the Ganges Basin",
  p.year           = 2022,
  p.abstract       = "This study presents a near-real-time flood mapping approach using Sentinel-1 GRD data over the Ganges basin in Bangladesh.",
  p.near_real_time = true,
  p.latency        = "6 h",
  p.revisit_time   = "6 days",
  p.confidence     = 0.90;

// 5b. Author
MERGE (a:Author {name: "Smith, John"})
MERGE (p)-[:AUTHORED_BY]->(a);

MERGE (a2:Author {name: "Jones, Maria"})
MERGE (p)-[:AUTHORED_BY]->(a2);

// 5c. DOI
MERGE (d:DOI {value: "10.1016/j.rse.2022.112345"})
MERGE (p)-[:HAS_DOI]->(d);

// 5d. Task type
MATCH (tt:TaskType {name: "Satellite flood mapping"})
MERGE (p)-[:HAS_TASK_TYPE]->(tt);

// 5e. Satellites and sensor type
MATCH (s1:Satellite {name: "Sentinel-1"})
MERGE (p)-[:USES_SATELLITE]->(s1);

MATCH (s2:Satellite {name: "Sentinel-2"})
MERGE (p)-[:USES_SATELLITE]->(s2);

// 5f. Data product
MATCH (dp:DataProduct {code: "GRD"})
MERGE (p)-[:USES_DATA_PRODUCT]->(dp);

// 5g. Methods
MATCH (m1:Method {name: "Thresholding"})
MERGE (p)-[:USES_METHOD]->(m1);

MATCH (m2:Method {name: "Change detection"})
MERGE (p)-[:USES_METHOD]->(m2);

MATCH (m3:Method {name: "NDWI/MNDWI"})
MERGE (p)-[:USES_METHOD]->(m3);

// 5h. Study area
MERGE (sa:StudyArea {country: "Bangladesh"})
SET sa.region = "South Asia", sa.river_basin = "Ganges"
WITH sa
MERGE (p)-[:STUDIES_AREA]->(sa);

// 5i. Event (optional)
MERGE (ev:Event {name: "2022 Bangladesh Floods"})
SET ev.year = 2022, ev.country = "Bangladesh"
WITH ev
MERGE (p)-[:ANALYZES_EVENT]->(ev);

// 5j. Metrics — OPTIONAL; omit entirely for review/hydraulic papers
MATCH (mt_oa:Metric   {type: "OA"})
MERGE (p)-[:REPORTS_METRIC {value: 0.92}]->(mt_oa);

MATCH (mt_f1:Metric   {type: "F1"})
MERGE (p)-[:REPORTS_METRIC {value: 0.88}]->(mt_f1);

MATCH (mt_iou:Metric  {type: "IoU"})
MERGE (p)-[:REPORTS_METRIC {value: 0.83}]->(mt_iou);


// ───────────────────────────────────────────────────────────────────────────────
// SECTION 5b — EXAMPLE: INSERT A REVIEW PAPER (no metrics, no satellite)
// ───────────────────────────────────────────────────────────────────────────────

MERGE (p2:Paper {source_file: "Kumar_2023_Review_DL_Flood.pdf"})
SET
  p2.title      = "Deep Learning Methods for Satellite-Based Flood Mapping: A Systematic Review",
  p2.year       = 2023,
  p2.confidence = 0.75;

MATCH (tt:TaskType {name: "Review paper"})
MERGE (p2)-[:HAS_TASK_TYPE]->(tt);

MERGE (a3:Author {name: "Kumar, Anil"})
MERGE (p2)-[:AUTHORED_BY]->(a3);

// No REPORTS_METRIC — valid because metrics are optional


// ───────────────────────────────────────────────────────────────────────────────
// SECTION 6 — ANALYTICAL CYPHER QUERIES
// ───────────────────────────────────────────────────────────────────────────────


// Q1 — Papers using SAR sensors, ordered by confidence
// -------------------------------------------------------
MATCH (p:Paper)-[:USES_SATELLITE]->(s:Satellite)-[:HAS_SENSOR_TYPE]->(st:SensorType {type: "SAR"})
RETURN p.title, p.year, collect(DISTINCT s.name) AS satellites, p.confidence
ORDER BY p.confidence DESC;


// Q2 — All papers studying Bangladesh with their methods
// -------------------------------------------------------
MATCH (p:Paper)-[:STUDIES_AREA]->(sa:StudyArea {country: "Bangladesh"})
OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
RETURN p.title, p.year, collect(DISTINCT m.name) AS methods
ORDER BY p.year DESC;


// Q3 — Best F1 scores reported across all papers
// -------------------------------------------------------
MATCH (p:Paper)-[r:REPORTS_METRIC]->(mt:Metric {type: "F1"})
RETURN p.title, p.year, r.value AS F1
ORDER BY F1 DESC
LIMIT 10;


// Q4 — Near-real-time papers: satellite + latency
// -------------------------------------------------------
MATCH (p:Paper {near_real_time: true})-[:USES_SATELLITE]->(s:Satellite)
RETURN p.title, p.year, collect(DISTINCT s.name) AS satellites,
       p.latency, p.revisit_time
ORDER BY p.year DESC;


// Q5 — Method frequency across all papers
// -------------------------------------------------------
MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
RETURN m.name, m.category, count(p) AS paper_count
ORDER BY paper_count DESC;


// Q6 — Most common method per sensor type
// -------------------------------------------------------
MATCH (p:Paper)-[:USES_SATELLITE]->(s:Satellite)-[:HAS_SENSOR_TYPE]->(st:SensorType),
      (p)-[:USES_METHOD]->(m:Method)
WITH st.type AS sensor, m.name AS method, count(p) AS cnt
ORDER BY sensor, cnt DESC
WITH sensor, collect({method: method, count: cnt})[0] AS top
RETURN sensor, top.method AS most_used_method, top.count AS paper_count;


// Q7 — Coverage per country (how many papers study each country)
// -------------------------------------------------------
MATCH (p:Paper)-[:STUDIES_AREA]->(sa:StudyArea)
RETURN sa.country AS country, sa.river_basin, count(p) AS paper_count
ORDER BY paper_count DESC;


// Q8 — Papers that report both F1 > 0.85 and use U-Net
// -------------------------------------------------------
MATCH (p:Paper)-[r:REPORTS_METRIC]->(mt:Metric {type: "F1"})
WHERE r.value > 0.85
MATCH (p)-[:USES_METHOD]->(m:Method {name: "U-Net"})
RETURN p.title, p.year, r.value AS F1;


// Q9 — Papers that have NO metrics (review / hydraulic / operational)
// -------------------------------------------------------
MATCH (p:Paper)
WHERE NOT (p)-[:REPORTS_METRIC]->()
MATCH (p)-[:HAS_TASK_TYPE]->(tt:TaskType)
RETURN p.title, p.year, tt.name AS task_type
ORDER BY tt.name, p.year;


// Q10 — Co-authorship: authors who published together
// -------------------------------------------------------
MATCH (a1:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(a2:Author)
WHERE id(a1) < id(a2)
RETURN a1.name, a2.name, count(p) AS shared_papers
ORDER BY shared_papers DESC
LIMIT 20;


// Q11 — Satellite usage trend by year
// -------------------------------------------------------
MATCH (p:Paper)-[:USES_SATELLITE]->(s:Satellite)
WHERE p.year IS NOT NULL
RETURN p.year AS year, s.name AS satellite, count(p) AS paper_count
ORDER BY year, paper_count DESC;


// Q12 — Average OA per method category
// -------------------------------------------------------
MATCH (p:Paper)-[:USES_METHOD]->(m:Method),
      (p)-[r:REPORTS_METRIC]->(mt:Metric {type: "OA"})
RETURN m.category, round(avg(r.value) * 1000) / 1000 AS avg_OA,
       count(p) AS n_papers
ORDER BY avg_OA DESC;


// Q13 — Full profile of one paper (all connected nodes)
// -------------------------------------------------------
MATCH (p:Paper {source_file: "Smith_2022_Bangladesh_SAR.pdf"})
OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
OPTIONAL MATCH (p)-[:HAS_DOI]->(d:DOI)
OPTIONAL MATCH (p)-[:HAS_TASK_TYPE]->(tt:TaskType)
OPTIONAL MATCH (p)-[:USES_SATELLITE]->(s:Satellite)-[:HAS_SENSOR_TYPE]->(st:SensorType)
OPTIONAL MATCH (p)-[:USES_DATA_PRODUCT]->(dp:DataProduct)
OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
OPTIONAL MATCH (p)-[:STUDIES_AREA]->(sa:StudyArea)
OPTIONAL MATCH (p)-[:ANALYZES_EVENT]->(ev:Event)
OPTIONAL MATCH (p)-[r:REPORTS_METRIC]->(mt:Metric)
RETURN
  p.title                           AS title,
  p.year                            AS year,
  collect(DISTINCT a.name)          AS authors,
  d.value                           AS doi,
  tt.name                           AS task_type,
  collect(DISTINCT s.name)          AS satellites,
  collect(DISTINCT st.type)         AS sensor_types,
  collect(DISTINCT dp.code)         AS data_products,
  collect(DISTINCT m.name)          AS methods,
  sa.country                        AS country,
  sa.river_basin                    AS river_basin,
  ev.name                           AS event,
  p.near_real_time                  AS near_real_time,
  p.latency                         AS latency,
  collect(DISTINCT {metric: mt.type, value: r.value}) AS metrics;
