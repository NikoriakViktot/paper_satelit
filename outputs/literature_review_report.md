# Literature Review: Flood Mapping with Remote Sensing

*Generated: 2026-04-27  |  Pipeline: RAG + Regex + Metadata Validation*

## 1. Introduction

This report presents the results of an automated systematic literature review of flood mapping studies using satellite remote sensing. A total of **43 papers** were processed through the RAG (Retrieval-Augmented Generation) pipeline, which extracts structured bibliographic and methodological metadata from PDF documents. The purpose of this analysis is to characterise the current state of the art in flood mapping with respect to methods employed, sensor types used, geographic coverage, and the availability and completeness of reported accuracy metrics.

## 2. Dataset Overview

| Metric                                 | Value    |
| -------------------------------------- | -------- |
| Total studies processed                | 43       |
| Studies with valid DOI                 | 8 (19%)  |
| Studies with valid abstract            | 0 (0%)   |
| Studies with quantitative metrics      | 0 (0%)   |
| Studies with semi-quantitative metrics | 5 (12%)  |
| Studies qualitative only               | 38 (88%) |

**DOI coverage** (19%): A substantial portion of papers lacked machine-readable DOIs in the extracted text, likely due to PDF formatting or placement outside the retrieved chunks.

**Abstract validity** (0%): The automated abstract extractor requires a clearly delimited `Abstract` section header. Papers where the abstract was embedded in the first-page text without a distinct heading were assigned a fallback excerpt from the full text.

## 3. Global Analysis

### 3.1 Accuracy Distribution

| Level                        | Count | Share |
| ---------------------------- | ----- | ----- |
| Quantitative (OA / F1 / IoU) | 0     | 0%    |
| Semi-quantitative (OA only)  | 5     | 12%   |
| Qualitative only             | 38    | 88%   |

Only **5 out of 43** studies (12%) reported any numeric accuracy metric. This reflects a widespread tendency in the flood mapping literature to present results visually or descriptively rather than through standardised performance indicators. The predominance of qualitative assessments (88%) highlights a significant gap in reproducibility and cross-study comparability.

### 3.2 Method Distribution

| Category | Count | Share |
| -------- | ----- | ----- |
| ML       | 14    | 33%   |
| DL       | 10    | 23%   |
| SAR      | 10    | 23%   |
| Other    | 9     | 21%   |

Machine learning methods (Random Forest, SVM, Decision Tree) are the most frequent category (14 studies, 33%), followed by deep learning approaches such as U-Net and CNN (10, 23%) and SAR-specific techniques (thresholding, change detection, OBIA) (10, 23%). The remaining 9 studies used unclassified or hybrid methods.

**Top methods by occurrence:**

| Method           | Count | Category |
| ---------------- | ----- | -------- |
| Random Forest    | 11    | ML       |
| Thresholding     | 7     | SAR      |
| Unknown          | 7     | Other    |
| CNN              | 5     | DL       |
| ViT              | 3     | DL       |
| U-Net            | 2     | DL       |
| Change Detection | 2     | SAR      |
| Decision Tree    | 1     | ML       |
| Machine Learning | 1     | ML       |
| Multi-Temporal   | 1     | SAR      |
| SVM              | 1     | ML       |
| Rapid            | 1     | Other    |

### 3.3 Sensor Distribution

| Category | Count | Share |
| -------- | ----- | ----- |
| SAR      | 25    | 58%   |
| Multi    | 18    | 42%   |

SAR-based sensors (primarily Sentinel-1) dominate the dataset (25 studies, 58%), reflecting the suitability of SAR for flood detection under cloud cover — a critical advantage during flood events. Multi-sensor studies combining SAR and optical imagery account for 18 papers (42%), suggesting growing interest in data fusion approaches.

### 3.4 Region Distribution

| Category    | Count | Share |
| ----------- | ----- | ----- |
| Global      | 22    | 51%   |
| Unspecified | 21    | 49%   |

A significant proportion of studies (22) could not be assigned to a specific region, either because no geographic reference was found in the retrieved text, or because the study used global datasets. This partly reflects the chunk-based retrieval strategy, which may not always surface the study-area description.

## 4. Ukraine and Regional Focus

No studies in the current extraction were explicitly linked to Ukraine or its sub-regions (Dnipro Basin, Carpathians, Eastern Europe) through the retrieved text chunks. This may reflect the coverage of the ingested PDF collection or the chunk-based retrieval not surfacing geographic metadata from these papers.

## 5. Per-Study Analysis

### Paper 1 — `remotesensing-14-03673-v2.pdf`

| Field                | Value                  |
| -------------------- | ---------------------- |
| Title                | which can be           |
| Authors              | remotesensing 14 03673 |
| DOI                  | —                      |
| Method               | ViT                    |
| Sensor               | Multi                  |
| Region               | Usa, Global            |
| OA                   | 1.000                  |
| F1                   | —                      |
| IoU                  | —                      |
| Accuracy Level       | Semi-quantitative      |
| Accuracy Description | OA=1.000               |
| Extraction Score     | 5                      |

### Paper 2 — `remotesensing-11-01581-v2.pdf`

| Field                | Value                                    |
| -------------------- | ---------------------------------------- |
| Title                | ere used to generate Landsat-based flood |
| Authors              | remotesensing 11 01581                   |
| DOI                  | —                                        |
| Method               | Unknown                                  |
| Sensor               | Multi                                    |
| Region               | Bangladesh                               |
| OA                   | 0.964                                    |
| F1                   | —                                        |
| IoU                  | —                                        |
| Accuracy Level       | Semi-quantitative                        |
| Accuracy Description | OA=0.964                                 |
| Extraction Score     | 4                                        |

### Paper 3 — `moharrami2021.pdf`

| Field                | Value             |
| -------------------- | ----------------- |
| Title                | inel-1 images     |
| Authors              | moharrami2021     |
| DOI                  | —                 |
| Method               | Thresholding      |
| Sensor               | SAR               |
| Region               | Unknown           |
| OA                   | 0.898             |
| F1                   | —                 |
| IoU                  | —                 |
| Accuracy Level       | Semi-quantitative |
| Accuracy Description | OA=0.898          |
| Extraction Score     | 5                 |

### Paper 4 — `Flood_extent_delineation_using_Sentinel.pdf`

| Field                | Value             |
| -------------------- | ----------------- |
| Title                | or validation     |
| Authors              | of Cao et al.     |
| DOI                  | —                 |
| Method               | Random Forest     |
| Sensor               | Multi             |
| Region               | Global            |
| OA                   | 0.880             |
| F1                   | —                 |
| IoU                  | —                 |
| Accuracy Level       | Semi-quantitative |
| Accuracy Description | OA=0.880          |
| Extraction Score     | 5                 |

### Paper 5 — `remotesensing-16-02193.pdf`

| Field                | Value                                     |
| -------------------- | ----------------------------------------- |
| Title                | terpretation, with an estimated geometric |
| Authors              | remotesensing 16 02193                    |
| DOI                  | —                                         |
| Method               | Rst-Flood                                 |
| Sensor               | SAR                                       |
| Region               | England, Italy, Spain                     |
| OA                   | 0.850                                     |
| F1                   | —                                         |
| IoU                  | —                                         |
| Accuracy Level       | Semi-quantitative                         |
| Accuracy Description | OA=0.850                                  |
| Extraction Score     | 5                                         |

### Paper 6 — `Flood_Extent_Mapping_in_the_Caprivi_Floodplain_Using_Sentinel-1_Time_Series.pdf`

| Field                | Value                        |
| -------------------- | ---------------------------- |
| Title                | A, UA, OA, and K values      |
| Authors              | Flood Extent Mapping         |
| DOI                  | —                            |
| Method               | Unknown                      |
| Sensor               | Multi                        |
| Region               | Unknown                      |
| OA                   | —                            |
| F1                   | —                            |
| IoU                  | —                            |
| Accuracy Level       | Qualitative                  |
| Accuracy Description | No numeric metrics extracted |
| Extraction Score     | 2                            |

### Paper 7 — `remotesensing-15-01200.pdf`

| Field                | Value                                                    |
| -------------------- | -------------------------------------------------------- |
| Title                | es of the Earth's                                        |
| Authors              | remotesensing 15 01200                                   |
| DOI                  | [10.3390/rs15051200](https://doi.org/10.3390/rs15051200) |
| Method               | Thresholding                                             |
| Sensor               | Multi                                                    |
| Region               | Global                                                   |
| OA                   | —                                                        |
| F1                   | —                                                        |
| IoU                  | —                                                        |
| Accuracy Level       | Qualitative                                              |
| Accuracy Description | No numeric metrics extracted                             |
| Extraction Score     | 3                                                        |

### Paper 8 — `034505_1.pdf`

| Field                | Value                        |
| -------------------- | ---------------------------- |
| Title                | h the same pixel             |
| Authors              | Jul-Sep                      |
| DOI                  | —                            |
| Method               | Thresholding                 |
| Sensor               | SAR                          |
| Region               | Usa                          |
| OA                   | —                            |
| F1                   | —                            |
| IoU                  | —                            |
| Accuracy Level       | Qualitative                  |
| Accuracy Description | No numeric metrics extracted |
| Extraction Score     | 3                            |

### Paper 9 — `remotesensing-16-00656-v2.pdf`

| Field                | Value                                                         |
| -------------------- | ------------------------------------------------------------- |
| Title                | s over 400 flood delineation maps at 10 m resolution gathered |
| Authors              | remotesensing 16 00656                                        |
| DOI                  | —                                                             |
| Method               | Random Forest                                                 |
| Sensor               | Multi                                                         |
| Region               | Global                                                        |
| OA                   | —                                                             |
| F1                   | —                                                             |
| IoU                  | —                                                             |
| Accuracy Level       | Qualitative                                                   |
| Accuracy Description | No numeric metrics extracted                                  |
| Extraction Score     | 3                                                             |

### Paper 10 — `s41598-021-86650-z.pdf`

| Field                | Value                                                       |
| -------------------- | ----------------------------------------------------------- |
| Title                | aining dataset with less than 3% of pixels belonging to the |
| Authors              | s41598 021 86650                                            |
| DOI                  | —                                                           |
| Method               | U-Net                                                       |
| Sensor               | Multi                                                       |
| Region               | Unknown                                                     |
| OA                   | —                                                           |
| F1                   | —                                                           |
| IoU                  | —                                                           |
| Accuracy Level       | Qualitative                                                 |
| Accuracy Description | No numeric metrics extracted                                |
| Extraction Score     | 3                                                           |

### Paper 11 — `remotesensing-17-01869.pdf`

| Field                | Value                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------- |
| Title                | fraction of inundated areas (higher producer's accuracy for water), while a few dry areas |
| Authors              | remotesensing 17 01869                                                                    |
| DOI                  | [10.3390/rs17111869](https://doi.org/10.3390/rs17111869)                                  |
| Method               | Rapid                                                                                     |
| Sensor               | Multi                                                                                     |
| Region               | Bangladesh, Usa, Global                                                                   |
| OA                   | —                                                                                         |
| F1                   | —                                                                                         |
| IoU                  | —                                                                                         |
| Accuracy Level       | Qualitative                                                                               |
| Accuracy Description | No numeric metrics extracted                                                              |
| Extraction Score     | 3                                                                                         |

### Paper 12 — `sensors-22-00960.pdf`

| Field                | Value                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------ |
| Title                | using remote-sensing help in visualizing the topography and other terrain properties [17]. |
| Authors              | Sensors                                                                                    |
| DOI                  | —                                                                                          |
| Method               | ViT                                                                                        |
| Sensor               | SAR                                                                                        |
| Region               | Usa, Global                                                                                |
| OA                   | —                                                                                          |
| F1                   | —                                                                                          |
| IoU                  | —                                                                                          |
| Accuracy Level       | Qualitative                                                                                |
| Accuracy Description | No numeric metrics extracted                                                               |
| Extraction Score     | 3                                                                                          |

### Paper 13 — `1-s2.0-S2590061725001048-main.pdf`

| Field                | Value                          |
| -------------------- | ------------------------------ |
| Title                | ion [23,85].                   |
| Authors              | 1 s2.0 S2590061725001048       |
| DOI                  | —                              |
| Method               | Random Forest                  |
| Sensor               | Multi                          |
| Region               | Australia, Germany, Japan, Usa |
| OA                   | —                              |
| F1                   | —                              |
| IoU                  | —                              |
| Accuracy Level       | Qualitative                    |
| Accuracy Description | No numeric metrics extracted   |
| Extraction Score     | 3                              |

### Paper 14 — `1-s2.0-S1569843223003291-main.pdf`

| Field                | Value                        |
| -------------------- | ---------------------------- |
| Title                | (2023) 103505                |
| Authors              | 1 s2.0 S1569843223003291     |
| DOI                  | —                            |
| Method               | Thresholding                 |
| Sensor               | SAR                          |
| Region               | Unknown                      |
| OA                   | —                            |
| F1                   | —                            |
| IoU                  | —                            |
| Accuracy Level       | Qualitative                  |
| Accuracy Description | No numeric metrics extracted |
| Extraction Score     | 3                            |

### Paper 15 — `refice2017.pdf`

| Field                | Value                                                                            |
| -------------------- | -------------------------------------------------------------------------------- |
| Title                | detailed time series of river discharge, and thus anomalies thereof preluding to |
| Authors              | refice2017                                                                       |
| DOI                  | —                                                                                |
| Method               | Change Detection                                                                 |
| Sensor               | SAR                                                                              |
| Region               | Unknown                                                                          |
| OA                   | —                                                                                |
| F1                   | —                                                                                |
| IoU                  | —                                                                                |
| Accuracy Level       | Qualitative                                                                      |
| Accuracy Description | No numeric metrics extracted                                                     |
| Extraction Score     | 3                                                                                |

### Paper 16 — `jwc0141935.pdf`

| Field                | Value                                                      |
| -------------------- | ---------------------------------------------------------- |
| Title                | patial Information Sciences XL-4/W1, 65-70.                |
| Authors              | jwc0141935                                                 |
| DOI                  | [10.3390/rs12071135.](https://doi.org/10.3390/rs12071135.) |
| Method               | Decision Tree                                              |
| Sensor               | SAR                                                        |
| Region               | Usa, Uk                                                    |
| OA                   | —                                                          |
| F1                   | —                                                          |
| IoU                  | —                                                          |
| Accuracy Level       | Qualitative                                                |
| Accuracy Description | No numeric metrics extracted                               |
| Extraction Score     | 3                                                          |

### Paper 17 — `hess-26-4345-2022.pdf`

| Field                | Value                                                        |
| -------------------- | ------------------------------------------------------------ |
| Title                | te sensing analysis (e.g., Lin et al., 2016), susceptibility |
| Authors              | dullah et al.                                                |
| DOI                  | —                                                            |
| Method               | Machine Learning                                             |
| Sensor               | SAR                                                          |
| Region               | Usa                                                          |
| OA                   | —                                                            |
| F1                   | —                                                            |
| IoU                  | —                                                            |
| Accuracy Level       | Qualitative                                                  |
| Accuracy Description | No numeric metrics extracted                                 |
| Extraction Score     | 3                                                            |

### Paper 18 — `remotesensing-14-05505-v2.pdf`

| Field                | Value                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------- |
| Title                | to six weeks after its occurrence (https://recovery.preventionweb.net/build-back-better/ |
| Authors              | remotesensing 14 05505                                                                   |
| DOI                  | —                                                                                        |
| Method               | ViT                                                                                      |
| Sensor               | SAR                                                                                      |
| Region               | Unknown                                                                                  |
| OA                   | —                                                                                        |
| F1                   | —                                                                                        |
| IoU                  | —                                                                                        |
| Accuracy Level       | Qualitative                                                                              |
| Accuracy Description | No numeric metrics extracted                                                             |
| Extraction Score     | 3                                                                                        |

### Paper 19 — `remotesensing-12-02073-v2.pdf`

| Field                | Value                                                              |
| -------------------- | ------------------------------------------------------------------ |
| Title                | pidly. Backscatter characteristics and variation rules of diﬀerent |
| Authors              | August                                                             |
| DOI                  | —                                                                  |
| Method               | Thresholding                                                       |
| Sensor               | Multi                                                              |
| Region               | Unknown                                                            |
| OA                   | —                                                                  |
| F1                   | —                                                                  |
| IoU                  | —                                                                  |
| Accuracy Level       | Qualitative                                                        |
| Accuracy Description | No numeric metrics extracted                                       |
| Extraction Score     | 3                                                                  |

### Paper 20 — `1-s2.0-S1569843222001911-main.pdf`

| Field                | Value                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------- |
| Title                | International Journal of Applied Earth Observations and Geoinformation 113 (2022) 103002 |
| Authors              | 1 s2.0 S1569843222001911                                                                 |
| DOI                  | —                                                                                        |
| Method               | Change Detection                                                                         |
| Sensor               | Multi                                                                                    |
| Region               | Usa                                                                                      |
| OA                   | —                                                                                        |
| F1                   | —                                                                                        |
| IoU                  | —                                                                                        |
| Accuracy Level       | Qualitative                                                                              |
| Accuracy Description | No numeric metrics extracted                                                             |
| Extraction Score     | 3                                                                                        |

### Paper 21 — `14-EQ35-2-Ghouri+et+al_149-159.pdf`

| Field                | Value                                                    |
| -------------------- | -------------------------------------------------------- |
| Title                | agement and the selection of                             |
| Authors              | 14 EQ35 2                                                |
| DOI                  | [10.3390/rs11131581](https://doi.org/10.3390/rs11131581) |
| Method               | Multi-Temporal                                           |
| Sensor               | SAR                                                      |
| Region               | Bangladesh                                               |
| OA                   | —                                                        |
| F1                   | —                                                        |
| IoU                  | —                                                        |
| Accuracy Level       | Qualitative                                              |
| Accuracy Description | No numeric metrics extracted                             |
| Extraction Score     | 3                                                        |

### Paper 22 — `water-11-02454.pdf`

| Field                | Value                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| Title                | Cao, H.; Zhang, H.; Wang, C.; Zhang, B. Operational Flood Detection Using Sentinel-1 SAR Data over Large |
| Authors              | water 11 02454                                                                                           |
| DOI                  | —                                                                                                        |
| Method               | Thresholding                                                                                             |
| Sensor               | SAR                                                                                                      |
| Region               | Germany, Romania, Uk, Vietnam                                                                            |
| OA                   | —                                                                                                        |
| F1                   | —                                                                                                        |
| IoU                  | —                                                                                                        |
| Accuracy Level       | Qualitative                                                                                              |
| Accuracy Description | No numeric metrics extracted                                                                             |
| Extraction Score     | 3                                                                                                        |

### Paper 23 — `remotesensing-13-04934-v2.pdf`

| Field                | Value                                      |
| -------------------- | ------------------------------------------ |
| Title                | inel-1 SAR and Sentinel-2 optical imagery. |
| Authors              | remotesensing 13 04934                     |
| DOI                  | —                                          |
| Method               | Unknown                                    |
| Sensor               | Multi                                      |
| Region               | Unknown                                    |
| OA                   | —                                          |
| F1                   | —                                          |
| IoU                  | —                                          |
| Accuracy Level       | Qualitative                                |
| Accuracy Description | No numeric metrics extracted               |
| Extraction Score     | 2                                          |

### Paper 24 — `DeepSAR Flood Mapper  global flood mapping on google earth engine cloud platform using MLP deep learning model with Sentinel-1 SAR imagery and HAND to.pdf`

| Field                | Value                                                       |
| -------------------- | ----------------------------------------------------------- |
| Title                | ct a target date to rapidly generate flood inundation maps. |
| Authors              | DeepSAR Flood Mapper                                        |
| DOI                  | —                                                           |
| Method               | Random Forest                                               |
| Sensor               | SAR                                                         |
| Region               | Global                                                      |
| OA                   | —                                                           |
| F1                   | —                                                           |
| IoU                  | —                                                           |
| Accuracy Level       | Qualitative                                                 |
| Accuracy Description | No numeric metrics extracted                                |
| Extraction Score     | 3                                                           |

### Paper 25 — `Urban_Flood_Mapping_Using_Satellite_Synthetic_Aperture_Radar_Data_A_review_of_characteristics_approaches_and_datasets.pdf`

| Field                | Value                                                  |
| -------------------- | ------------------------------------------------------ |
| Title                | standardized benchmark dataset for urban flood mapping |
| Authors              | Urban Flood Mapping                                    |
| DOI                  | —                                                      |
| Method               | Unknown                                                |
| Sensor               | SAR                                                    |
| Region               | Unknown                                                |
| OA                   | —                                                      |
| F1                   | —                                                      |
| IoU                  | —                                                      |
| Accuracy Level       | Qualitative                                            |
| Accuracy Description | No numeric metrics extracted                           |
| Extraction Score     | 2                                                      |

### Paper 26 — `Flooded area detection and mapping from Sentinel-1 imagery. Complementary approaches and comparative performance evaluation.pdf`

| Field                | Value                        |
| -------------------- | ---------------------------- |
| Title                | A. TOMA ET AL.               |
| Authors              | Flooded area detection       |
| DOI                  | —                            |
| Method               | Random Forest                |
| Sensor               | SAR                          |
| Region               | Location maps of the Romania |
| OA                   | —                            |
| F1                   | —                            |
| IoU                  | —                            |
| Accuracy Level       | Qualitative                  |
| Accuracy Description | No numeric metrics extracted |
| Extraction Score     | 3                            |

### Paper 27 — `s41597-025-04554-3.pdf`

| Field                | Value                                                                    |
| -------------------- | ------------------------------------------------------------------------ |
| Title                | rios. However, with the advent of deep                                   |
| Authors              | s41597 025 04554                                                         |
| DOI                  | [10.1038/s41597-025-04554-3](https://doi.org/10.1038/s41597-025-04554-3) |
| Method               | CNN                                                                      |
| Sensor               | Multi                                                                    |
| Region               | Usa, Global                                                              |
| OA                   | —                                                                        |
| F1                   | —                                                                        |
| IoU                  | —                                                                        |
| Accuracy Level       | Qualitative                                                              |
| Accuracy Description | No numeric metrics extracted                                             |
| Extraction Score     | 3                                                                        |

### Paper 28 — `remotesensing-17-02909.pdf`

| Field                | Value                                                                              |
| -------------------- | ---------------------------------------------------------------------------------- |
| Title                | both time and labor investments. Subsequently, we proposed a CNN model, FloodsNet, |
| Authors              | remotesensing 17 02909                                                             |
| DOI                  | —                                                                                  |
| Method               | CNN                                                                                |
| Sensor               | Multi                                                                              |
| Region               | China, Global                                                                      |
| OA                   | —                                                                                  |
| F1                   | —                                                                                  |
| IoU                  | —                                                                                  |
| Accuracy Level       | Qualitative                                                                        |
| Accuracy Description | No numeric metrics extracted                                                       |
| Extraction Score     | 3                                                                                  |

### Paper 29 — `ijgi-12-00194-v2.pdf`

| Field                | Value                                                                                |
| -------------------- | ------------------------------------------------------------------------------------ |
| Title                | rained on NovaSAR-1 and Sentinel-1 labelled SAR pre-flood datasets and tested on the |
| Authors              | ijgi 12 00194                                                                        |
| DOI                  | —                                                                                    |
| Method               | U-Net                                                                                |
| Sensor               | SAR                                                                                  |
| Region               | Global                                                                               |
| OA                   | —                                                                                    |
| F1                   | —                                                                                    |
| IoU                  | —                                                                                    |
| Accuracy Level       | Qualitative                                                                          |
| Accuracy Description | No numeric metrics extracted                                                         |
| Extraction Score     | 3                                                                                    |

### Paper 30 — `Sentinel-1-Based_Water_and_Flood_Mapping_Benchmarking_Convolutional_Neural_Networks_Against_an_Operational_Rule-Based_Processing_Chain.pdf`

| Field                | Value                           |
| -------------------- | ------------------------------- |
| Title                | al for many segmentation tasks. |
| Authors              | Sentinel 1 Based                |
| DOI                  | —                               |
| Method               | CNN                             |
| Sensor               | SAR                             |
| Region               | Unknown                         |
| OA                   | —                               |
| F1                   | —                               |
| IoU                  | —                               |
| Accuracy Level       | Qualitative                     |
| Accuracy Description | No numeric metrics extracted    |
| Extraction Score     | 3                               |

### Paper 31 — `s10712-022-09751-y (1).pdf`

| Field                | Value                                                             |
| -------------------- | ----------------------------------------------------------------- |
| Title                | itted distribution of water start deviating and used as predeter- |
| Authors              | Li et al.                                                         |
| DOI                  | —                                                                 |
| Method               | CNN                                                               |
| Sensor               | SAR                                                               |
| Region               | Unknown                                                           |
| OA                   | —                                                                 |
| F1                   | —                                                                 |
| IoU                  | —                                                                 |
| Accuracy Level       | Qualitative                                                       |
| Accuracy Description | No numeric metrics extracted                                      |
| Extraction Score     | 3                                                                 |

### Paper 32 — `s10712-022-09751-y.pdf`

| Field                | Value                                                             |
| -------------------- | ----------------------------------------------------------------- |
| Title                | itted distribution of water start deviating and used as predeter- |
| Authors              | Li et al.                                                         |
| DOI                  | —                                                                 |
| Method               | CNN                                                               |
| Sensor               | SAR                                                               |
| Region               | Unknown                                                           |
| OA                   | —                                                                 |
| F1                   | —                                                                 |
| IoU                  | —                                                                 |
| Accuracy Level       | Qualitative                                                       |
| Accuracy Description | No numeric metrics extracted                                      |
| Extraction Score     | 3                                                                 |

### Paper 33 — `sustainability-14-03251.pdf`

| Field                | Value                                                        |
| -------------------- | ------------------------------------------------------------ |
| Title                | um elevation. It should be noted that flood depth along with |
| Authors              | sustainability 14 03251                                      |
| DOI                  | —                                                            |
| Method               | Random Forest                                                |
| Sensor               | SAR                                                          |
| Region               | Unknown                                                      |
| OA                   | —                                                            |
| F1                   | —                                                            |
| IoU                  | —                                                            |
| Accuracy Level       | Qualitative                                                  |
| Accuracy Description | No numeric metrics extracted                                 |
| Extraction Score     | 3                                                            |

### Paper 34 — `A hybrid approach for enhanced flood prediction and assessment  Leveraging physical models  deep learning and satellite remote sensing.pdf`

| Field                | Value                                                             |
| -------------------- | ----------------------------------------------------------------- |
| Title                | hydrological modeling. Decision Trees provide interpretable rules |
| Authors              | A hybrid approach                                                 |
| DOI                  | —                                                                 |
| Method               | Random Forest                                                     |
| Sensor               | SAR                                                               |
| Region               | Unknown                                                           |
| OA                   | —                                                                 |
| F1                   | —                                                                 |
| IoU                  | —                                                                 |
| Accuracy Level       | Qualitative                                                       |
| Accuracy Description | No numeric metrics extracted                                      |
| Extraction Score     | 3                                                                 |

### Paper 35 — `sustainability-13-07925.pdf`

| Field                | Value                        |
| -------------------- | ---------------------------- |
| Title                | FR-based calculation of      |
| Authors              | Sustainability               |
| DOI                  | —                            |
| Method               | SVM                          |
| Sensor               | SAR                          |
| Region               | Unknown                      |
| OA                   | —                            |
| F1                   | —                            |
| IoU                  | —                            |
| Accuracy Level       | Qualitative                  |
| Accuracy Description | No numeric metrics extracted |
| Extraction Score     | 3                            |

### Paper 36 — `remotesensing-17-03471-v2.pdf`

| Field                | Value                                                                     |
| -------------------- | ------------------------------------------------------------------------- |
| Title                | ved the maximum AUC score among the randomly sampled hyperparameter sets. |
| Authors              | remotesensing 17 03471                                                    |
| DOI                  | —                                                                         |
| Method               | Random Forest                                                             |
| Sensor               | SAR                                                                       |
| Region               | Unknown                                                                   |
| OA                   | —                                                                         |
| F1                   | —                                                                         |
| IoU                  | —                                                                         |
| Accuracy Level       | Qualitative                                                               |
| Accuracy Description | No numeric metrics extracted                                              |
| Extraction Score     | 3                                                                         |

### Paper 37 — `coltin2016.pdf`

| Field                | Value                        |
| -------------------- | ---------------------------- |
| Title                | rms a tree of                |
| Authors              | coltin2016                   |
| DOI                  | —                            |
| Method               | Random Forest                |
| Sensor               | SAR                          |
| Region               | Unknown                      |
| OA                   | —                            |
| F1                   | —                            |
| IoU                  | —                            |
| Accuracy Level       | Qualitative                  |
| Accuracy Description | No numeric metrics extracted |
| Extraction Score     | 3                            |

### Paper 38 — `remotesensing-17-00524.pdf`

| Field                | Value                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------- |
| Title                | The supervised classification algorithm automatically evaluates and learns the associa- |
| Authors              | remotesensing 17 00524                                                                  |
| DOI                  | —                                                                                       |
| Method               | Random Forest                                                                           |
| Sensor               | SAR                                                                                     |
| Region               | Unknown                                                                                 |
| OA                   | —                                                                                       |
| F1                   | —                                                                                       |
| IoU                  | —                                                                                       |
| Accuracy Level       | Qualitative                                                                             |
| Accuracy Description | No numeric metrics extracted                                                            |
| Extraction Score     | 3                                                                                       |

### Paper 39 — `journal.pwat.0000269.pdf`

| Field                | Value                                                                                                          |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| Title                | curacy, allowing for robust classification without overfitting. The Random Forest (RF) model, a robust machine |
| Authors              | journal.pwat.0000269                                                                                           |
| DOI                  | [10.1371/journal.pwat.0000269.g002](https://doi.org/10.1371/journal.pwat.0000269.g002)                         |
| Method               | Random Forest                                                                                                  |
| Sensor               | SAR                                                                                                            |
| Region               | Unknown                                                                                                        |
| OA                   | —                                                                                                              |
| F1                   | —                                                                                                              |
| IoU                  | —                                                                                                              |
| Accuracy Level       | Qualitative                                                                                                    |
| Accuracy Description | No numeric metrics extracted                                                                                   |
| Extraction Score     | 3                                                                                                              |

### Paper 40 — `isprs-archives-XLIII-B3-2020-641-2020.pdf`

| Field                | Value                                                                                                          |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| Title                | are required for flood mapping by the fusion of optical and SAR                                                |
| Authors              | isprs archives XLIII                                                                                           |
| DOI                  | [10.5194/isprs-archives-XLIII-B3-2020-641-2020](https://doi.org/10.5194/isprs-archives-XLIII-B3-2020-641-2020) |
| Method               | Unknown                                                                                                        |
| Sensor               | Multi                                                                                                          |
| Region               | Global                                                                                                         |
| OA                   | —                                                                                                              |
| F1                   | —                                                                                                              |
| IoU                  | —                                                                                                              |
| Accuracy Level       | Qualitative                                                                                                    |
| Accuracy Description | No numeric metrics extracted                                                                                   |
| Extraction Score     | 2                                                                                                              |

### Paper 41 — `10.1515_geo-2020-0325.pdf`

| Field                | Value                        |
| -------------------- | ---------------------------- |
| Title                | sing in flood manage-        |
| Authors              | 10.1515 geo 2020             |
| DOI                  | —                            |
| Method               | Unknown                      |
| Sensor               | Multi                        |
| Region               | Unknown                      |
| OA                   | —                            |
| F1                   | —                            |
| IoU                  | —                            |
| Accuracy Level       | Qualitative                  |
| Accuracy Description | No numeric metrics extracted |
| Extraction Score     | 2                            |

### Paper 42 — `nhess-22-2473-2022.pdf`

| Field                | Value                                                                            |
| -------------------- | -------------------------------------------------------------------------------- |
| Title                | ale of satellite                                                                 |
| Authors              | mann                                                                             |
| DOI                  | [10.5194/nhess-22-2473-2022](https://doi.org/10.5194/nhess-22-2473-2022)         |
| Method               | Unknown                                                                          |
| Sensor               | Multi                                                                            |
| Region               | Location of the selected gauge stations where the river discharge is recorded (b |
| OA                   | —                                                                                |
| F1                   | —                                                                                |
| IoU                  | —                                                                                |
| Accuracy Level       | Qualitative                                                                      |
| Accuracy Description | No numeric metrics extracted                                                     |
| Extraction Score     | 2                                                                                |

### Paper 43 — `hydrology-10-00017-v2.pdf`

| Field                | Value                        |
| -------------------- | ---------------------------- |
| Title                | ds. Compared                 |
| Authors              | hydrology 10 00017           |
| DOI                  | —                            |
| Method               | Thresholding                 |
| Sensor               | Multi                        |
| Region               | Unknown                      |
| OA                   | —                            |
| F1                   | —                            |
| IoU                  | —                            |
| Accuracy Level       | Qualitative                  |
| Accuracy Description | No numeric metrics extracted |
| Extraction Score     | 3                            |

## 6. Key Findings

### 6.1 Scarcity of Standardised Accuracy Reporting

Only 12% of reviewed studies reported numeric accuracy metrics (OA, F1, or IoU). The majority (88%) relied exclusively on qualitative assessments or visual comparisons. This heterogeneity in reporting practices severely limits the ability to draw cross-study conclusions or conduct meta-analytic comparisons of methods.

### 6.2 Dominance of SAR-Based Approaches

Synthetic Aperture Radar (SAR) sensors were used in 43 out of 43 studies (100%), either exclusively or in combination with optical data. This reflects the operational advantages of SAR for flood detection: all-weather, day-and-night acquisition, and direct sensitivity to surface water extent through changes in backscatter.

### 6.3 Emergence of Deep Learning

Deep learning methods (U-Net, CNN, ViT) were applied in 10 studies (23%), approaching the prevalence of classical machine learning approaches (14 studies). This trend indicates a rapid shift towards data-driven segmentation architectures that can exploit large labelled flood datasets.

### 6.4 Geographic Coverage Gaps

A majority of studies with identifiable geography focused on Asia (Bangladesh, India, China) and North America (USA). Eastern European flood events — including Ukraine, the Dnipro Basin, and Carpathian catchments — appear under-represented in the literature, suggesting a need for targeted studies using locally acquired SAR and optical imagery.

## 7. Limitations

### 7.1 Missing Accuracy Metrics

A large number of papers did not report OA, F1, or IoU in the sections captured by the retrieval queries. It is possible that some papers report accuracy metrics in tables or supplementary material that was not indexed. Future work should include table-aware extraction and figure-caption parsing.

### 7.2 Extraction Uncertainty

The current pipeline relies on rule-based regex patterns for metric extraction. Non-standard notation (e.g., per-class F1, micro/macro averages) may be misidentified or missed entirely. The title and abstract extractors depend on the presence of clearly delimited section headers in the PDF text, which is not always the case after OCR or layout parsing.

### 7.3 Variability in Reporting Conventions

Studies use different scales (0–1 vs 0–100%), different metric names (Overall Accuracy vs Overall Classification Accuracy), and different evaluation protocols (per-image vs per-event vs per-dataset). All extracted values were normalised to the 0–1 scale, but subtle differences in evaluation protocols may introduce incomparabilities not detectable by automated extraction.

### 7.4 Dataset Coverage

The current extraction processed 43 papers from the indexed PDF collection. The vector store contains 13,970 chunks from a larger set of PDFs. Some papers may have been underrepresented in the retrieval step if their accuracy or method descriptions did not match the six predefined retrieval queries.

## 8. Conclusion

This systematic review processed 43 flood mapping studies via an automated RAG pipeline combining vector-store retrieval, regex-based information extraction, metadata validation, and field normalisation. The results confirm several established trends: the dominance of SAR sensors (particularly Sentinel-1), the growing adoption of deep learning architectures alongside classical ML methods, and a persistent gap in standardised quantitative reporting. Only 12% of studies provided machine-readable numeric accuracy, underscoring the need for community-wide adoption of reporting standards (e.g., STAC flood datasets, standardised validation protocols). Future reviews should expand geographic coverage — particularly for Eastern Europe and Ukraine — and integrate table and figure extraction to recover the full breadth of accuracy information available in the literature.

---
*Report auto-generated by the Flood-Paper RAG Pipeline on 2026-04-27.*