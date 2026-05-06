import sys
import subprocess
import re
import json
import hashlib
import urllib.parse
from pathlib import Path
from lxml import etree
import spacy
import requests
import time
from sklearn.metrics.pairwise import cosine_similarity
import os
import traceback
import numpy as np


os.environ["HF_HOME"] = "/home/viktornikoriak/paper_satelit/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/viktornikoriak/paper_satelit/.hf_cache"

BASE = Path("/home/viktornikoriak/paper_satelit/data/literature")
XML_DIR = BASE / "grobid_xml"
OUT_DIR = BASE / "paper_json"

OUT_DIR.mkdir(exist_ok=True)

NS = {"tei": "http://www.tei-c.org/ns/1.0"}
GEO_CACHE = {}
LAST_CALL = 0

SATELLITE_PATTERNS = {
    "Sentinel-1": r"\bsentinel[\s\-]?1[abc]?\b|\bs1[abc]?\b",
    "Sentinel-2": r"\bsentinel[\s\-]?2[abc]?\b|\bs2[abc]?\b",
    "Landsat": r"\blandsat[\s\-]?(4|5|7|8|9)?\b",
    "MODIS": r"\bmodis\b",
    "VIIRS": r"\bviirs\b",
    "RADARSAT": r"\bradarsat[\s\-]?\d?\b",
    "TerraSAR-X": r"\bterrasar[\s\-]?x\b",
    "COSMO-SkyMed": r"\bcosmo[\s\-]?skymed\b",
    "ALOS PALSAR": r"\balos\b|\bpalsar\b",
    "PlanetScope": r"\bplanetscope\b|\bplanet\s+labs?\b",
    "WorldView": r"\bworldview[\s\-]?\d?\b",
    "Pleiades": r"\bpl[eé]iades\b",
    "ICESat-2": r"\bicesat[\s\-]?2\b|\batl0[368]\b",
}

DEM_DATASETS = {
    "SRTM": r"\bsrtm\b",
    "ASTER GDEM": r"\baster\b|\bgdem\b",
    "ALOS AW3D": r"\balos\b.*\baw3d\b|\baw3d\b",
    "TanDEM-X": r"\btandem[\-\s]?x\b",
    "Copernicus DEM": r"\bcopernicus dem\b|\bcopernicus digital elevation\b",
    "FABDEM": r"\bfabdem\b",
    "MERIT DEM": r"\bmerit\b",
    "NASADEM": r"\bnasadem\b",
    "LiDAR": r"\blidar\b",
}
METHOD_PATTERNS = {
    # =========================
    # Spectral / vegetation / water / drought indices
    # =========================
    "NDVI": r"\bndvi\b|normalized difference vegetation index",
    "EVI": r"\bevi\b|enhanced vegetation index",
    "EVI2": r"\bevi2\b|two[-\s]?band enhanced vegetation index",
    "SAVI": r"\bsavi\b|soil adjusted vegetation index",
    "MSAVI": r"\bmsavi\b|modified soil adjusted vegetation index",
    "OSAVI": r"\bosavi\b|optimized soil adjusted vegetation index",
    "DVI": r"\bdvi\b|difference vegetation index",
    "GVI": r"\bgvi\b|green vegetation index",
    "MTVI": r"\bmtvi\b|modified transformed vegetation index",

    "NDWI": r"\bndwi\b|normalized difference water index",
    "MNDWI": r"\bmndwi\b|modified normalized difference water index",
    "AWEI": r"\bawei\b|automated water extraction index",
    "LSWI": r"\blswi\b|land surface water index",
    "NDMI": r"\bndmi\b|normalized difference moisture index",
    "NDII": r"\bndii\b|normalized difference infrared index",
    "NWI": r"\bnwi\b|normalized water index",
    "DSWI": r"\bdswi\b|disease water stress index|drought stress water index",
    "RDI": r"\brdi\b|ratio drought index|relative drought index",
    "MSI": r"\bmsi\b|moisture stress index",
    "SR-SWIR": r"\bsr[-\s]?swir\b|simple ratio swir",
    "TVDI": r"\btvdi\b|temperature vegetation dryness index",
    "VCI": r"\bvci\b|vegetation condition index",
    "TCI": r"\btci\b|temperature condition index",
    "VHI": r"\bvhi\b|vegetation health index",

    "Spectral indices": (
        r"\bspectral indices?\b|"
        r"\bvegetation indices?\b|"
        r"\bwater indices?\b|"
        r"\bmoisture indices?\b|"
        r"\bdrought indices?\b"
    ),

    # =========================
    # SAR flood mapping / radar methods
    # =========================
    "Backscatter analysis": r"\bbackscatter(?:ing)?\b|sigma0|sigma nought|sigma[-\s]?0",
    "SAR backscatter thresholding": r"\bsar\b.*\bthreshold(?:ing)?\b|\bthreshold(?:ing)?\b.*\bsar\b",
    "VV/VH ratio": r"\bvv\s*/\s*vh\b|\bvv[-\s]?vh ratio\b|\bpolarization ratio\b",
    "Polarization analysis": r"\bpolarization\b|\bpolarisation\b|\bvv\b|\bvh\b|\bhh\b|\bhv\b",
    "SAR change detection": r"\bsar\b.*\bchange detection\b|\bchange detection\b.*\bsar\b",
    "Log-ratio change detection": r"\blog[-\s]?ratio\b|\blog ratio\b",
    "Image differencing": r"\bimage differenc(?:e|ing)\b|\bdifference image\b",
    "Pre-post flood comparison": r"\bpre[-\s]?flood\b.*\bpost[-\s]?flood\b|\bpre and post\b",
    "NDSI SAR": r"\bndsi\b|normalized difference scattering index",
    "SNDSI": r"\bsndsi\b|shannon.*ndsi|entropy.*ndsi",
    "Bayesian inference": r"\bbayes\b|\bbayesian\b|bayes inference",
    "Robust Satellite Technique": r"\brst\b|robust satellite technique|sar[-\s]?rst",
    "Interferometric coherence": r"\bcoherence\b|interferometric coherence|insar coherence",
    "SAR polarimetry": r"\bpolarimetric\b|\bpolarimetry\b|polsar",
    "Object-based SAR classification": r"\bobject[-\s]?based\b.*\bsar\b|\bsar\b.*\bobia\b",

    # =========================
    # Thresholding / segmentation
    # =========================
    "Thresholding": r"\bthreshold(?:ing)?\b",
    "Otsu threshold": r"\botsu\b|otsu'?s method",
    "Kittler-Illingworth threshold": r"\bkittler\b|\bki method\b|kittler[-\s]?illingworth",
    "Adaptive threshold": r"\badaptive threshold(?:ing)?\b",
    "Histogram threshold": r"\bhistogram threshold(?:ing)?\b",
    "Bimodal threshold": r"\bbimodal threshold(?:ing)?\b",
    "Region growing": r"\bregion growing\b",
    "Watershed segmentation": r"\bwatershed segmentation\b",
    "K-means clustering": r"\bk[-\s]?means\b|kmeans",
    "Fuzzy C-means": r"\bfuzzy c[-\s]?means\b|\bfcm\b",
    "Mean shift": r"\bmean shift\b",
    "Superpixel segmentation": r"\bsuperpixel\b|slic",

    # =========================
    # General remote sensing change detection
    # =========================
    "Change detection": r"\bchange detection\b|multi[-\s]?temporal|bitemporal|bi[-\s]?temporal",
    "Post-classification comparison": r"\bpost[-\s]?classification comparison\b|\bpcc\b",
    "Change vector analysis": r"\bchange vector analysis\b|\bcva\b",
    "Time-series analysis": r"\btime[-\s]?series\b|temporal analysis",
    "Trend analysis": r"\btrend analysis\b|temporal trend",
    "Mann-Kendall test": r"\bmann[-\s]?kendall\b|\bmk test\b",
    "Sen slope": r"\bsen'?s slope\b|theil[-\s]?sen",
    "BFAST": r"\bbfast\b|breaks for additive season and trend",
    "LandTrendr": r"\blandtrendr\b",
    "CCDC": r"\bccdc\b|continuous change detection and classification",

    # =========================
    # Classical machine learning
    # =========================
    "Random Forest": r"\brandom forest\b|\brf classifier\b|\brf\b",
    "SVM": r"\bsvm\b|support vector machine",
    "Decision Tree": r"\bdecision tree\b|\bdt classifier\b",
    "XGBoost": r"\bxgboost\b|extreme gradient boosting",
    "LightGBM": r"\blightgbm\b",
    "CatBoost": r"\bcatboost\b",
    "Gradient Boosting": r"\bgradient boosting\b|\bgbm\b",
    "AdaBoost": r"\badaboost\b|adaptive boosting",
    "Naive Bayes": r"\bnaive bayes\b",
    "Logistic Regression": r"\blogistic regression\b",
    "KNN": r"\bknn\b|k[-\s]?nearest neighbors?",
    "Maximum likelihood": r"\bmaximum likelihood\b|\bmlc\b",
    "Minimum distance": r"\bminimum distance\b",
    "Mahalanobis distance": r"\bmahalanobis\b",
    "OBIA": r"\bobia\b|object[-\s]?based image analysis|object[-\s]?based classification",

    # =========================
    # Deep learning / computer vision
    # =========================
    "CNN": r"\bcnn\b|convolutional neural network",
    "FCN": r"\bfcn\b|fully convolutional network",
    "U-Net": r"\bu[-\s]?net\b|unet",
    "ResNet": r"\bresnet\b|residual network",
    "DeepLab": r"\bdeeplab\b|deeplabv3\+?",
    "SegNet": r"\bsegnet\b",
    "YOLO": r"\byolo\b|yolov\d+",
    "Vision Transformer": r"\bvision transformer\b|\bvit\b",
    "Transformer": r"\btransformer\b",
    "LSTM": r"\blstm\b|long short[-\s]?term memory",
    "GRU": r"\bgru\b|gated recurrent unit",
    "Autoencoder": r"\bautoencoder\b",
    "GAN": r"\bgan\b|generative adversarial network",
    "Attention mechanism": r"\battention mechanism\b|\bself[-\s]?attention\b",
    "Knowledge distillation": r"\bknowledge distillation\b",
    "Transfer learning": r"\btransfer learning\b",
    "Self-supervised learning": r"\bself[-\s]?supervised\b",
    "Quantization-aware training": r"\bquantization[-\s]?aware training\b|\bqat\b",
    "Model pruning": r"\bstructured pruning\b|\bmodel pruning\b|\bpruning\b",
    "Edge deployment": r"\bedge deployment\b|edge computing",

    # =========================
    # Hydrological / hydraulic models
    # =========================
    "HEC-RAS": r"\bhec[-\s]?ras\b",
    "HEC-HMS": r"\bhec[-\s]?hms\b",
    "SWAT": r"\bswat\b|soil and water assessment tool",
    "SWAT+": r"\bswat\+\b|swat plus",
    "WRF-Hydro": r"\bwrf[-\s]?hydro\b",
    "LISFLOOD": r"\blisflood\b",
    "MIKE FLOOD": r"\bmike flood\b",
    "MIKE 11": r"\bmike\s?11\b",
    "MIKE 21": r"\bmike\s?21\b",
    "FLO-2D": r"\bflo[-\s]?2d\b",
    "TOPKAPI": r"\btopkapi\b",
    "Xinanjiang": r"\bxinanjiang\b",
    "HBV": r"\bhbv\b|hydrologiska byr[aå]ns vattenbalansavdelning",
    "GR4J": r"\bgr4j\b",
    "SCS-CN": r"\bscs[-\s]?cn\b|curve number|runoff curve number",
    "Unit hydrograph": r"\bunit hydrograph\b",
    "Rainfall-runoff modeling": r"\brainfall[-\s]?runoff\b|runoff modeling|runoff simulation",
    "Hydrological modelling": r"\bhydrological modell?ing\b|hydrological simulation",
    "Hydraulic modelling": r"\bhydraulic modell?ing\b|hydrodynamic modell?ing\b|2d hydraulic",
    "Flood frequency analysis": r"\bflood frequency analysis\b|return period|gumbel distribution|gev distribution",
    "Extreme value analysis": r"\bextreme value analysis\b|\bgev\b|generalized extreme value",

    # =========================
    # DEM / terrain / geomorphometry
    # =========================
    "DEM analysis": r"\bdem analysis\b|digital elevation model analysis|terrain analysis",
    "DEM validation": r"\bdem validation\b|vertical accuracy|elevation accuracy",
    "HAND": r"\bhand\b|height above nearest drainage|height above nearest stream",
    "TWI": r"\btwi\b|topographic wetness index",
    "SPI terrain": r"\bstream power index\b",
    "LS factor": r"\bls factor\b|slope length",
    "Slope analysis": r"\bslope analysis\b|\bslope\b",
    "Aspect analysis": r"\baspect\b",
    "Curvature analysis": r"\bcurvature\b|profile curvature|plan curvature",
    "Flow direction": r"\bflow direction\b|\bd8\b|d[-\s]?infinity|dinf",
    "Flow accumulation": r"\bflow accumulation\b",
    "Sink filling": r"\bsink fill(?:ing)?\b|fill sinks|depression filling",
    "Depression breaching": r"\bdepression breach(?:ing)?\b|breach depressions",
    "Watershed delineation": r"\bwatershed delineation\b|basin delineation|catchment delineation",
    "Stream extraction": r"\bstream extraction\b|drainage extraction|channel extraction",

    # =========================
    # Remote sensing preprocessing
    # =========================
    "Radiometric calibration": r"\bradiometric calibration\b",
    "Atmospheric correction": r"\batmospheric correction\b",
    "Topographic correction": r"\btopographic correction\b",
    "Terrain correction": r"\bterrain correction\b|range doppler terrain correction",
    "Geometric correction": r"\bgeometric correction\b|orthorectification",
    "Speckle filtering": r"\bspeckle filter(?:ing)?\b",
    "Lee filter": r"\blee filter\b|refined lee",
    "Frost filter": r"\bfrost filter\b",
    "Gamma-MAP filter": r"\bgamma[-\s]?map\b",
    "Median filtering": r"\bmedian filter(?:ing)?\b",
    "Cloud masking": r"\bcloud mask(?:ing)?\b|cloud removal",
    "Mosaicking": r"\bmosaic(?:king)?\b",
    "Image compositing": r"\bcomposite\b|image compositing",
    "Band math": r"\bband math\b|bandmaths|raster calculator",
    "Layer stacking": r"\blayer stack(?:ing)?\b",
    "Subsetting": r"\bsubset(?:ting)?\b|clip(?:ping)?",
    "Reprojection": r"\breproject(?:ion)?\b|coordinate transformation",
    "Resampling": r"\bresampl(?:e|ing)\b",
    "HDF conversion": r"\bhdf\b.*\bconversion\b|convert.*hdf",

    # =========================
    # GIS / spatial analysis
    # =========================
    "GIS": r"\bgis\b|geographic information system",
    "QGIS": r"\bqgis\b",
    "ArcGIS": r"\barcgis\b",
    "ERDAS Imagine": r"\berdas\b|erdas imagine",
    "SNAP": r"\bsnap\b|sentinel application platform",
    "Google Earth Engine": r"\bgoogle earth engine\b|\bgee\b",
    "Zonal statistics": r"\bzonal statistics\b",
    "Buffer analysis": r"\bbuffer analysis\b|buffer zone",
    "Overlay analysis": r"\boverlay analysis\b|spatial overlay",
    "Spatial interpolation": r"\bspatial interpolation\b|kriging|idw interpolation",
    "Kriging": r"\bkriging\b|ordinary kriging|universal kriging",
    "IDW": r"\bidw\b|inverse distance weighting",
    "Raster classification": r"\braster classification\b|reclassification",
    "Map algebra": r"\bmap algebra\b",

    # =========================
    # Statistical analysis / validation
    # =========================
    "Regression analysis": r"\bregression analysis\b|\blinear regression\b|multiple regression",
    "Correlation analysis": r"\bcorrelation\b|pearson correlation|spearman correlation",
    "Principal component analysis": r"\bpca\b|principal component analysis",
    "Accuracy assessment": r"\baccuracy assessment\b|confusion matrix",
    "Confusion matrix": r"\bconfusion matrix\b",
    "Cross-validation": r"\bcross[-\s]?validation\b|k[-\s]?fold",
    "Sensitivity analysis": r"\bsensitivity analysis\b",
    "Uncertainty analysis": r"\buncertainty analysis\b|error propagation",
    "AUC ROC": r"\bauc\b|\broc\b|receiver operating characteristic",

    # =========================
    # Flood / water body extraction named workflows
    # =========================
    "ADWB": r"\badwb\b|automatic detection of water bodies",
    "Water body extraction": r"\bwater body extraction\b|surface water extraction",
    "Flood extent mapping": r"\bflood extent\b|flooded area extraction|inundation extent",
    "Flood damage assessment": r"\bflood damage assessment\b|damage assessment",
    "Flood susceptibility mapping": r"\bflood susceptibility\b|flood susceptibility mapping",
    "Flood hazard mapping": r"\bflood hazard mapping\b",
    "Flood risk mapping": r"\bflood risk mapping\b",

    "Water indices": r"\bwater indices?\b|\bmoisture indices?\b",
    "MODIS MOD13C2": r"\bmod13c2\b",
    "Meteorological observations": r"\bmeteorological observations?\b|\bweather stations?\b",
    "Projection": r"\butm\b|\bwgs84\b|\bprojection\b|\breproject(?:ion)?\b",
    "Spatial Modeler": r"\bspatial modeler\b|\bmodel maker\b",
}

METHOD_PATTERNS["HAND"] = (
    r"\bhand\s+(?:index|model|method|mask)\b|"
    r"height above nearest drainage|"
    r"height above nearest stream"
)

METHOD_PATTERNS.update({
    "Change detection": r"\bchange detection\b",
    "Double scattering": r"\bdouble scattering\b",
    "Backscatter analysis": r"\bbackscatter\b",
    "Edge detection": r"\bsobel\b|\broberts\b",
    "Region growing": r"\bregion growing\b",
    "HASARD": r"\bhasard\b",
    "SAR intensity analysis": r"\bsar intensity\b",
})

METRIC_PATTERNS = {
    "OA": r"(overall accuracy|global accuracy|\boa\b)\s*(?:=|:|of|values?)?\s*(\d+(?:\.\d+)?)\s*%?",
    "F1": r"(f1[\-\s]?score|\bf1\b|f[\-\s]?measure)\s*(?:=|:|of|values?)?\s*(\d+(?:\.\d+)?)\s*%?",
    "IoU": r"(\biou\b|intersection over union|jaccard)\s*(?:=|:|of|values?)?\s*(\d+(?:\.\d+)?)\s*%?",
    "Kappa": r"(kappa|kp)\s*(?:coefficient)?\s*(?:=|:|of|values?)?\s*(\d+(?:\.\d+)?)\s*%?",
    "RMSE": r"\brmse\b\s*(?:=|:|of)?\s*(\d+(?:\.\d+)?)",
    "MAE": r"\bmae\b\s*(?:=|:|of)?\s*(\d+(?:\.\d+)?)",
    "R": r"\br\s*=\s*(\d+(?:\.\d+)?)",
    "p_value": r"\bp\s*[<=>]\s*(\d+(?:\.\d+)?)",
    "Percent": r"(up to|around|approximately|about)?\s*(\d+(?:\.\d+)?)\s*%",
}

COUNTRY_PATTERNS = {
    "Ukraine": r"\bukraine\b|\bukrainian\b|\bkyiv\b|\bkherson\b|\bkakhovka\b|\bzakarpattia\b|\bcarpathians?\b",
    "Poland": r"\bpoland\b|\bpolish\b",
    "Germany": r"\bgermany\b|\bgerman\b",
    "France": r"\bfrance\b|\bfrench\b",
    "Italy": r"\bitaly\b|\bitalian\b",
    "Spain": r"\bspain\b|\bspanish\b",
    "Portugal": r"\bportugal\b|\bportuguese\b",
    "Netherlands": r"\bnetherlands\b|\bdutch\b",
    "Belgium": r"\bbelgium\b",
    "Austria": r"\baustria\b",
    "Switzerland": r"\bswitzerland\b",
    "Slovenia": r"\bslovenia\b",
    "Croatia": r"\bcroatia\b",
    "Hungary": r"\bhungary\b",
    "Czech Republic": r"\bczech\b",
    "Slovakia": r"\bslovakia\b",
    "Romania": r"\bromania\b",
    "Bulgaria": r"\bbulgaria\b",
    "Greece": r"\bgreece\b",
    "Turkey": r"\bturkey\b",
    "UK": r"\buk\b|united kingdom|england|scotland|wales",
    "Ireland": r"\bireland\b",

    "USA": r"\busa\b|united states|louisiana|california|texas|florida",
    "Canada": r"\bcanada\b",
    "Mexico": r"\bmexico\b",
    "Brazil": r"\bbrazil\b",
    "Argentina": r"\bargentina\b",
    "Chile": r"\bchile\b",
    "Peru": r"\bperu\b",

    "India": r"\bindia\b",
    "China": r"\bchina\b",
    "Japan": r"\bjapan\b",
    "South Korea": r"\bkorea\b",
    "Vietnam": r"\bvietnam\b",
    "Thailand": r"\bthailand\b",
    "Indonesia": r"\bindonesia\b",
    "Malaysia": r"\bmalaysia\b",
    "Pakistan": r"\bpakistan\b",
    "Bangladesh": r"\bbangladesh\b",
    "Uzbekistan": r"\buzbekistan\b",
    "Kazakhstan": r"\bkazakhstan\b",
    "Iran": r"\biran\b",
    "Iraq": r"\biraq\b",

    "Australia": r"\baustralia\b|new south wales|queensland",
    "New Zealand": r"\bnew zealand\b",

    "South Africa": r"\bsouth africa\b",
    "Egypt": r"\begypt\b",
    "Morocco": r"\bmorocco\b",
    "Madagascar": r"\bmadagascar\b",
    "Nigeria": r"\bnigeria\b",
}

COUNTRY_PATTERNS["Philippines"] = r"\bphilippines\b|\bluzon\b"
COUNTRY_PATTERNS["India"] = r"\bindia\b|\bkerala\b"

RIVER_TO_COUNTRY = {
    # Ukraine
    "Dnipro": "Ukraine",
    "Dnieper": "Ukraine",
    "Prut": "Ukraine/Romania",
    "Dniester": "Ukraine/Moldova",
    "Tisza": "Ukraine/Hungary",
    "Southern Bug": "Ukraine",
    "Desna": "Ukraine",
    "Moshchunka": "Ukraine",

    # Europe
    "Danube": "Multiple",
    "Rhine": "Germany/France/Netherlands",
    "Elbe": "Germany/Czech Republic",
    "Seine": "France",
    "Loire": "France",
    "Thames": "UK",
    "Po": "Italy",
    "Ebro": "Spain",
    "Tagus": "Spain/Portugal",
    "Duero": "Spain/Portugal",
    "Carrion": "Spain",
    "Krka": "Slovenia",
    "Sava": "Slovenia/Croatia",
    "Drava": "Austria/Slovenia/Croatia",
    "Vistula": "Poland",
    "Oder": "Germany/Poland",

    # Asia
    "Ganges": "India/Bangladesh",
    "Indus": "India/Pakistan",
    "Yangtze": "China",
    "Yellow River": "China",
    "Mekong": "Multiple",
    "Irrawaddy": "Myanmar",

    # Americas
    "Mississippi": "USA",
    "Colorado": "USA",
    "Amazon": "Brazil/Peru",
    "Orinoco": "Venezuela",
    "Parana": "Argentina/Brazil",

    # Africa
    "Nile": "Multiple",
    "Congo": "DR Congo",
    "Niger": "Multiple",
    "Zambezi": "Multiple",

    # Australia
    "Darling": "Australia",
    "Murray": "Australia",
}

RIVER_PATTERNS = [
    r"\b([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){0,3})\s+River basin\b",
    r"\b([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){0,3})\s+River\b",
    r"\b(Dnipro|Dnieper|Prut|Dniester|Danube|Tisza|Moshchunka|Krka|Sava|Darling|Carrion|Duero|Indus|Ganges)\b",
]

RIVER_TO_COUNTRY.update({
    "Cagayan": "Philippines",
    "Cagayan River": "Philippines",
})

INVALID_LOCATIONS = {
    "earth", "world", "globe", "surface",
    "region", "area", "study", "model",
    "figure", "table", "section", "introduction",
    "results", "discussion"
}

INVALID_LOCATIONS.update({
    "dsm",
    "worlddem",
    "sentinel",
    "sentinel-1",
    "sentinel-2",
    "sar",
    "gee",
    "google earth engine",
    "north america",
    "western europe",
    "the middle east",
    "middle east",
})

STUDY_TYPE_PROTOTYPES = {
    "case_study": [
        "study conducted in a specific river basin",
        "analysis of a flood event in a region",
        "case study of a watershed"
    ],
    "multi_site": [
        "evaluation across multiple locations",
        "analysis on many flood events",
        "tested on various regions"
    ],
    "global_algorithmic": [
        "global flood mapping algorithm",
        "method applied worldwide",
        "large scale automatic system"
    ],
    "regional": [
        "analysis across a country",
        "study using multiple stations in one country",
        "regional climate analysis"
    ],
    # 🔥 НОВЕ
    "review": [
        "this paper reviews existing methods",
        "we survey the literature",
        "systematic review of flood mapping techniques",
        "comparison of different approaches",
        "literature review on remote sensing"
    ]
}

TASK_PROTOTYPES = {
    "flood_mapping_satellite": [
        "flood mapping using Sentinel-1 SAR imagery",
        "water extent extraction using satellite data",
        "flood detection using remote sensing",
        "inundation mapping using SAR data",
        "satellite based flood monitoring",
        "flood extent mapping from SAR or optical satellite imagery",
    ],

    "flood_modeling_hydraulic": [
        "flood simulation using HEC-RAS",
        "2D hydraulic flood modeling",
        "hydrodynamic flood simulation",
        "river flow modeling using hydraulic equations",
        "flood depth and velocity simulation",
        "inundation modeling using hydraulic models",
    ],

    "hydrological_modeling": [
        "rainfall runoff modeling using SWAT",
        "hydrological simulation using HEC-HMS",
        "basin scale runoff modeling",
        "precipitation discharge modeling",
        "streamflow simulation",
        "catchment runoff forecasting",
    ],

    "spectral_index_analysis": [
        "NDVI NDWI NDII RDI spectral index analysis",
        "water and vegetation indices for land surface monitoring",
        "remote sensing indices for soil moisture and drought",
        "vegetation and moisture index analysis",
        "spectral indices for land surface analysis",
        "water index based surface condition monitoring",
    ],

    "drought_monitoring": [
        "drought monitoring using MODIS vegetation indices",
        "soil moisture assessment using NDII and RDI",
        "desertification monitoring using remote sensing",
        "climate change impact on terrestrial ecosystems",
        "aridity assessment using satellite indices",
        "soil moisture and vegetation stress monitoring",
    ],

    "land_cover_classification": [
        "land cover classification using remote sensing",
        "image classification using machine learning",
        "supervised classification of satellite imagery",
        "LULC classification using optical satellite data",
    ],

    "land_use_change_detection": [
        "land use land cover change detection",
        "land cover transition analysis using satellite imagery",
        "ecosystem change detection using remote sensing",
        "land use change monitoring in disaster affected regions",
        "vegetation and forest cover dynamics",
    ],

    "flood_damage_assessment": [
        "flood damage assessment using satellite imagery",
        "agricultural loss assessment after flood",
        "infrastructure damage mapping after flood",
        "rapid flood damage estimation using SAR data",
    ],

    "flood_susceptibility_mapping": [
        "flood susceptibility mapping using machine learning",
        "flood hazard mapping using topographic and hydrological factors",
        "urban flood susceptibility analysis",
        "flood risk mapping using GIS and remote sensing factors",
    ],

    "terrain_dem_analysis": [
        "digital elevation model analysis",
        "terrain analysis using DEM",
        "slope aspect elevation modeling",
        "topographic analysis for flood modeling",
        "geomorphometric analysis using elevation data",
    ],

    "dem_validation": [
        "digital elevation model validation using ICESat-2",
        "DEM vertical accuracy assessment",
        "terrain model comparison using reference elevation data",
        "elevation error assessment using LiDAR or ICESat-2",
    ],

    "review": [
        "review of flood mapping methods",
        "survey of remote sensing approaches",
        "overview of flood detection techniques",
        "literature review on flood monitoring",
        "systematic review of flood susceptibility mapping",
    ],
    "unknown": [
        "unknown task",
        "insufficient evidence to classify the scientific task",
    ],
}

REGION_PATTERNS = {
    "Luzon": r"\bluzon\b",
    "Cagayan Valley": r"\bcagayan valley\b",
    "Northern Philippines": r"\bnorthern philippines\b",

    "Steppe zone": r"\bsteppe zone\b",
    "Forest-steppe zone": r"\bforest[-\s]?steppe\b",
    "Ukrainian Carpathians": r"\bukrainian carpathians\b|\bcarpathians\b",
    "Crimean Mountains": r"\bcrimean mountains\b",
    "Kyiv Oblast": r"\bkyiv oblast\b",
    "Bucha district": r"\bbucha district\b",
    "Zakarpattia": r"\bzakarpattia\b|\btranscarpathia\b",
    "Kherson": r"\bkherson\b",
    "Krka floodplain": r"\bkrka river floodplain\b|\bkrka floodplain\b",
    "Lower Krka": r"\blower krka\b",
    "Krakovo forest": r"\bkrakovo forest\b",
}

REGION_PATTERNS.update({
    "Kerala": r"\bkerala\b",
    "Fishlake": r"\bfishlake\b",
    "Pontypridd": r"\bpontypridd\b",
    "Rhondda Cynon Taf": r"\brhondda cynon taf\b",
})

VALID_TASK_LABELS = set(TASK_PROTOTYPES)

VALID_STUDY_TYPES = {
    "case_study",
    "regional",
    "multi_site",
    "global_algorithmic",
    "review",
    "unknown",
}

INLINE_HEADINGS = [
    ("introduction", r"\bIntroduction\.\s+"),
    ("methods", r"\bMaterials and methods\.\s+"),
    ("methods", r"\bMaterial and methods\.\s+"),
    ("methods", r"\bMethods\.\s+"),
    ("results", r"\bResults(?: and discussion)?\.\s+"),
    ("discussion", r"\bDiscussion\.\s+"),
    ("conclusion", r"\bConclusions?\.\s+"),
]

GENERIC_REGION_NOISE = {
    "specific region",
    "study region",
    "target region",
    "selected region",
    "this region",
    "the region",
    "region",
    "area",
}

GENERIC_REGION_NOISE.update({
    "geographical region",
    "geographical regions",
    "selected locations",
    "selected regions",
    "region a",
    "region b",
    "region c",
})

INVALID_METRIC_CONTEXT = [
    "aep",
    "annual exceedance probability",
    "world settlement footprint",
    "wsf",
    "table",
]

class PipelineContext:
    def __init__(self, sections: dict):
        self.sections = ensure_dict(sections)
        self.abstract = self.sections.get("abstract", "")
        self.introduction = self.sections.get("introduction", "")
        self.study_area = self.sections.get("study_area", "")
        self.data_sources = self.sections.get("data_sources", "")
        self.methods = self.sections.get("methods", "")
        self.results = self.sections.get("results", "")
        self.conclusion = self.sections.get("conclusion", "")
        self.other = self.sections.get("other", "")

        self.full_text = build_full_text(self.sections)

        self.study_country_text = " ".join([
            self.abstract,
            self.study_area,
            self.data_sources,
            self.methods,
            self.other,
        ])

        self.satellite_text = " ".join([
            self.abstract,
            self.data_sources,
            self.methods,
            self.other,
        ])

        self.method_text = " ".join([
            self.abstract,
            self.data_sources,
            self.methods,
            self.results,
            self.other,
        ])

        self.metric_text = " ".join([
            self.abstract,
            self.results,
            self.conclusion,
        ])

def is_valid_metric(snippet):
    snippet = snippet.lower()
    return not any(k in snippet for k in INVALID_METRIC_CONTEXT)

def best_mention_context(name: str, text: str, window: int = 500) -> str:
    if not name or not text:
        return ""

    matches = list(re.finditer(re.escape(name), text, flags=re.I))

    if not matches:
        return ""

    scored = []

    for m in matches[:10]:
        ctx = snippet(text, m.start(), m.end(), window=window)
        score = 0

        ctx_l = ctx.lower()

        for k in [
            "study area",
            "study site",
            "tested on",
            "test site",
            "flood event",
            "flood events",
            "occurred in",
            "located in",
            "case study",
            "data sources",
            "ground truth",
            "validation data",
        ]:
            if k in ctx_l:
                score += 1

        scored.append((score, ctx))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def make_task(label: str, confidence: float, source: str = "rules") -> dict:
    if label not in VALID_TASK_LABELS:
        raise ValueError(f"Unknown task label: {label}")

    return {
        "label": label,
        "confidence": float(confidence),
        "source": source,
    }

def default_study_geo() -> dict:
    return {
        "primary_country": None,
        "countries": [],
        "regions": [],
        "rivers": [],
        "river_country_links": [],
        "locations": [],
        "coordinates": [],
        "confidence": 0.0,
    }

def ensure_dict(value, default=None):
    if isinstance(value, dict):
        return value
    return default if default is not None else {}

def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    if isinstance(obj, set):
        return [json_safe(v) for v in obj]

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, Path):
        return str(obj)

    return obj

def add_context_score(entity, ctx_text):
    if not entity["name"] or not ctx_text:
        return entity

    ctx = best_mention_context(entity["name"], ctx_text)

    score = 0
    ctx_l = ctx.lower()

    if any(k in ctx_l for k in [
        "study area", "case study", "located in",
        "study site", "basin"
    ]):
        score += 0.5

    if any(k in ctx_l for k in [
        "used", "applied", "analysis", "simulation"
    ]):
        score += 0.3

    entity["scores"]["context"] = score
    entity["evidence"] = ctx or entity["evidence"]

    return entity

def detect_role(entity, ctx_text):
    ctx = best_mention_context(entity["name"], ctx_text)
    ctx_l = ctx.lower()

    if any(k in ctx_l for k in [
        "used", "applied", "model", "analysis",
        "derived from", "calculated"
    ]):
        entity["role"] = "used"
    else:
        entity["role"] = "mentioned"

    return entity

class OllamaJudge:
    def __init__(self, base_url="http://localhost:11434", model="llama3.1:8b", timeout=120):
        self.url = f"{base_url.rstrip('/')}/api/generate"
        self.model = model
        self.timeout = timeout

    def judge(self, candidate_json: dict, sections: dict) -> dict:
        prompt = self._build_prompt(candidate_json, sections)

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "top_p": 0.2
                }
            },
            timeout=self.timeout
        )
        response.raise_for_status()

        raw = response.json().get("response", "")
        return self._parse_json(raw)

    def _parse_json(self, text: str) -> dict:
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError(f"Ollama returned non-JSON output: {text[:500]}")
            return json.loads(match.group(0))

    def _build_prompt(self, candidate_json: dict, sections: dict) -> str:
        title = candidate_json.get("title") or sections.get("title") or ""
        abstract = sections.get("abstract") or ""
        introduction = sections.get("introduction") or ""
        methods = "\n".join([
            sections.get("methods", ""),
            sections.get("study_area", ""),
            sections.get("data_sources", ""),
        ])

        return f"""
    You are a scientific knowledge graph judge.

    Your task is NOT to extract new information.
    Your task is to validate extracted candidate facts against evidence from the paper.

    Return valid JSON only.
    Do NOT include markdown.
    Do NOT include explanations outside JSON.

    Allowed study_type labels ONLY:
    - case_study
    - regional
    - multi_site
    - global_algorithmic
    - review
    - unknown

    Allowed task labels ONLY:
    - flood_mapping_satellite
    - flood_modeling_hydraulic
    - hydrological_modeling
    - spectral_index_analysis
    - drought_monitoring
    - land_cover_classification
    - land_use_change_detection
    - flood_damage_assessment
    - flood_susceptibility_mapping
    - terrain_dem_analysis
    - dem_validation
    - review
    - unknown

    Rules:
    1. Study country must come from study area/title/abstract/methods, not author affiliation.
    2. Rivers are accepted only if they are the actual study area.
    3. Rivers mentioned in citations or literature review must be rejected.
    4. Sentinel/Landsat/MODIS/RADARSAT are satellites.
    5. SRTM/ASTER GDEM/FABDEM/Copernicus DEM/MERIT DEM are DEM datasets.
    6. If the paper maps flood extent using Sentinel-1, Sentinel-2, SAR, backscatter, NDSI, SNDSI, Standardized Residuals, SR, Bayesian Inference, Otsu, KI, thresholding, CNN, ResNet, or satellite images, task MUST be flood_mapping_satellite.
    7. Do NOT classify flood mapping as drought_monitoring.
    8. drought_monitoring is only valid when drought, aridity, desertification, soil moisture, NDII, RDI, MODIS vegetation/moisture indices are central.
    9. hydrological_modeling means rainfall-runoff, discharge, SWAT, HEC-HMS, streamflow simulation.
    10. flood_modeling_hydraulic means HEC-RAS, LISFLOOD, MIKE, 2D hydrodynamic modeling.
    11. review means systematic review, literature review, or overview paper.
    12. If accepted=true, corrected_value MUST be null or equal to original_value.
    13. If corrected_value differs from original_value, accepted MUST be false.
    14. If accepted=false, corrected_value MUST NOT be null.
    15. If unsure, use "unknown".

    INPUT CANDIDATES:
    {json.dumps(candidate_json, ensure_ascii=False, indent=2)}

    EVIDENCE:
    TITLE:
    {title[:1000]}

    ABSTRACT:
    {abstract[:2500]}

    INTRODUCTION:
    {introduction[:2500]}

    METHODS / STUDY AREA / DATA:
    {methods[:3500]}

    Return JSON with this schema:
    {{
      "paper_id": "...",
      "study_type": {{
        "accepted": true,
        "original_value": "...",
        "corrected_value": null,
        "confidence": 0.0,
        "reason": "...",
        "evidence": "..."
      }},
      "study_country": {{
        "accepted": true,
        "original_value": "...",
        "corrected_value": null,
        "confidence": 0.0,
        "reason": "...",
        "evidence": "..."
      }},
      "rivers": [],
      "data_sources": [],
      "task": {{
        "accepted": true,
        "original_value": "...",
        "corrected_value": null,
        "confidence": 0.0,
        "reason": "...",
        "evidence": "..."
      }}
    }}
    """

def load_embedding_model():
    import os
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer

    # 🔥 завантажує .env
    load_dotenv()

    # 🔥 тепер змінна доступна
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment")

    # 🔥 задати кеш локально (правильно)
    os.environ["HF_HOME"] = "./.hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "./.hf_cache"

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        use_auth_token=hf_token  # 🔥 головне
    )

model = load_embedding_model()

def classify_with_embeddings(text: str):
    text = text or ""

    text_emb = model.encode([text])[0]
    scores = {}

    for label, examples in STUDY_TYPE_PROTOTYPES.items():
        example_embs = model.encode(examples)
        sim = cosine_similarity([text_emb], example_embs).mean()
        scores[label] = float(sim)

    best = max(scores, key=scores.get)

    return best, scores

def classify_with_embeddings_ctx(ctx: PipelineContext):
    return classify_with_embeddings(ctx.full_text)

def classify_task_with_embeddings(text: str):
    text = text or ""

    text_emb = model.encode([text])[0]
    scores = {}

    for label, examples in TASK_PROTOTYPES.items():
        example_embs = model.encode(examples)
        sim = cosine_similarity([text_emb], example_embs).mean()
        scores[label] = float(sim)

    best = max(scores, key=scores.get)

    return best, scores

def embedding_score(text: str, label_examples: list):
    text_emb = model.encode([text])[0]
    ex_embs = model.encode(label_examples)

    sims = cosine_similarity([text_emb], ex_embs)[0]

    return float(max(sims))  # 🔥 НЕ mean

def is_valid_place(name: str) -> bool:
    if not name:
        return False

    n = name.strip().lower()

    if len(n) < 3:
        return False

    if n in INVALID_LOCATIONS:
        return False

    if any(k in n for k in ["earth", "world", "surface"]):
        return False

    return True

def load_spacy(model_name: str = "en_core_web_md"):
    try:
        return spacy.load(model_name)

    except OSError:
        print(f"⚠️ spaCy model '{model_name}' not found. Installing...")

        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True
        )

        return spacy.load(model_name)

nlp = load_spacy("en_core_web_trf")

def extract_geo_ner(text: str):
    doc = nlp(text)

    countries = set()
    locations = set()

    for ent in doc.ents:
        name = ent.text.strip()

        if not is_valid_place(name):
            continue

        if ent.label_ == "GPE":
            countries.add(name)

        elif ent.label_ == "LOC":
            locations.add(name)

    return list(countries), list(locations)

def classify_location(name: str):
    n = name.lower()

    if "river" in n:
        return "river"

    if "lake" in n:
        return "lake"

    if any(k in n for k in ["basin", "catchment", "delta"]):
        return "basin"

    if any(k in n for k in ["watershed"]):
        return "watershed"

    if n in {
        "donbas", "carpathians", "balkans",
        "upper mekong", "lower danube"
    }:
        return "region"

    return "region"

def normalize_country_item(item):
    if isinstance(item, dict):
        name = clean_text(item.get("name"))
        if not name:
            return None

        return {
            **item,
            "name": name,
            "source": item.get("source", "unknown"),
            "confidence": float(item.get("confidence", 0.5)),
        }

    name = clean_text(str(item))
    if not name:
        return None

    return {
        "name": name,
        "code": None,
        "source": "ner",
        "confidence": 0.55,
    }


def merge_countries(existing, ner_countries):
    merged = {}

    for item in existing or []:
        c = normalize_country_item(item)
        if c:
            merged[c["name"]] = c

    for item in ner_countries or []:
        c = normalize_country_item(item)
        if not c:
            continue

        name = c["name"]

        if name not in merged:
            merged[name] = c
        else:
            merged[name]["confidence"] = max(
                float(merged[name].get("confidence", 0.5)),
                float(c.get("confidence", 0.55)),
            )

    return list(merged.values())


def compute_geo_confidence(study_geo: dict) -> float:
    geo = ensure_dict(study_geo)

    score = 0.0

    if geo.get("primary_country"):
        score += 0.35

    if geo.get("countries"):
        score += 0.15

    if geo.get("regions"):
        score += 0.20

    if geo.get("rivers"):
        score += 0.15

    if geo.get("locations"):
        score += 0.10

    if geo.get("coordinates"):
        score += 0.05

    return round(min(score, 1.0), 4)


def validate_with_ner(study_geo, ner_countries, ner_locations):
    validated = ensure_dict(study_geo, default_study_geo()).copy()

    validated.pop("study_geo", None)

    if ner_countries:
        validated["countries"] = merge_countries(
            validated.get("countries", []),
            ner_countries
        )

    validated_locations = []

    for loc in validated.get("locations", []):
        loc = ensure_dict(loc)

        if not loc:
            continue

        if loc.get("source") == "ner_study_area_context" and loc.get("evidence"):
            loc["validated"] = True
            loc["confidence"] = min(1.0, float(loc.get("confidence", 0.7)) + 0.1)
            validated_locations.append(loc)

    validated["locations"] = validated_locations
    validated["confidence"] = compute_geo_confidence(validated)

    return validated

def geonames_type(fcode: str):
    if not fcode:
        return "unknown"

    if fcode.startswith("H"):
        return "river"
    if fcode.startswith("P"):
        return "city"
    if fcode.startswith("A"):
        return "admin"
    if fcode.startswith("T"):
        return "terrain"

    return "other"

def geonames_lookup(name: str):
    global LAST_CALL

    if not name:
        return None

    # 🔥 rate limit
    delay = 1.0
    elapsed = time.time() - LAST_CALL

    if elapsed < delay:
        time.sleep(delay - elapsed)

    LAST_CALL = time.time()

    url = "http://api.geonames.org/searchJSON"

    params = {
        "q": name,
        "maxRows": 5,  # 🔥 беремо кілька варіантів
        "username": "viktornikoriak"
    }

    try:
        r = requests.get(url, params=params, timeout=5)

        # 🔥 перевірка статусу
        if r.status_code != 200:
            return None

        data = r.json()
        results = data.get("geonames", [])

        if not results:
            return None

        # 🔥 вибір найкращого результату
        best = None

        for g in results:
            fcode = g.get("fcode", "")

            # 🔥 фільтр нормальних гео-типів
            if fcode.startswith(("P", "A", "H", "T")):
                best = g
                break

        # fallback якщо не знайшли нормальний
        if not best:
            best = results[0]

        lat = best.get("lat")
        lon = best.get("lng")

        if lat is None or lon is None:
            return None

        return {
            "name": best.get("name"),
            "lat": float(lat),
            "lon": float(lon),
            "type": geonames_type(best.get("fcode")),
            "country": best.get("countryName"),
            "feature": best.get("fcodeName"),
            "feature_code": best.get("fcode"),
            "source": "geonames"
        }

    except Exception:
        return None

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def snippet(text: str, start: int, end: int, window: int = 120) -> str:
    return clean_text(text[max(0, start - window):min(len(text), end + window)])

def make_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def doi_to_url(doi: str | None) -> str | None:
    return f"https://doi.org/{doi}" if doi else None

def scholar_url(title: str | None) -> str | None:
    if not title:
        return None
    return "https://scholar.google.com/scholar?q=" + urllib.parse.quote(title)

def first_text(root, xpath: str) -> str | None:
    values = root.xpath(xpath, namespaces=NS)
    if not values:
        return None
    return clean_text(str(values[0]))

def all_text(node, xpath: str) -> str:
    return clean_text(" ".join(node.xpath(xpath, namespaces=NS)))

def build_full_text(sections):
    return " ".join([
        sections.get("abstract", ""),
        sections.get("introduction", ""),
        sections.get("study_area", ""),
        sections.get("data_sources", ""),
        sections.get("methods", ""),
        sections.get("results", ""),
        sections.get("other", ""),
    ])

def parse_authors(root) -> list[dict]:
    authors = []

    for a in root.xpath("//tei:sourceDesc//tei:analytic/tei:author", namespaces=NS):
        first = clean_text(" ".join(a.xpath(".//tei:forename/text()", namespaces=NS)))
        last = clean_text(" ".join(a.xpath(".//tei:surname/text()", namespaces=NS)))
        email = first_text(a, ".//tei:email/text()")
        orcid = first_text(a, ".//tei:idno[@type='ORCID']/text()")
        affiliation = clean_text(" ".join(a.xpath(".//tei:affiliation//text()", namespaces=NS)))

        countries = [
            normalize_country_name(c)
            for c in a.xpath(".//tei:country/text()", namespaces=NS)
        ]
        countries = [c for c in countries if c]

        full_name = clean_text(f"{first} {last}") if first or last else None

        if full_name:
            authors.append({
                "first_name": first or None,
                "last_name": last or None,
                "full_name": full_name,
                "email": email,
                "orcid": orcid,
                "affiliation": affiliation or None,
                "affiliation_countries": countries,
            })

    return authors

def author_name_set(metadata: dict) -> set[str]:
    names = set()

    for author in metadata.get("authors", []):
        for key in ["first_name", "last_name", "full_name"]:
            value = author.get(key)

            if not value:
                continue

            for part in str(value).split():
                p = clean_text(part).lower()
                if len(p) > 2:
                    names.add(p)

    return names

def extract_author_geo(metadata: dict) -> list[dict]:
    countries = {}

    for author in metadata.get("authors", []):
        for country in author.get("affiliation_countries", []):
            name = normalize_country_name(country)

            if name:
                countries[name] = {
                    "name": name,
                    "source": "tei_author_affiliation",
                    "confidence": 0.95,
                }

    return list(countries.values())

def parse_metadata(root, xml_path: Path) -> dict:
    title = (
        first_text(root, "//tei:titleStmt/tei:title/text()")
        or first_text(root, "//tei:sourceDesc//tei:analytic/tei:title/text()")
    )

    doi = first_text(root, "//tei:idno[@type='DOI']/text()")
    year = first_text(root, "//tei:date/@when")
    if year:
        year = year[:4]

    journal = first_text(root, "//tei:sourceDesc//tei:monogr/tei:title/text()")
    publisher = (
        first_text(root, "//tei:sourceDesc//tei:monogr//tei:publisher/text()")
        or first_text(root, "//tei:publicationStmt/tei:publisher/text()")
    )
    url = doi_to_url(doi) or first_text(root, "//tei:ptr/@target") or scholar_url(title)

    return {
        "paper_id": xml_path.stem.replace(".tei", ""),
        "source_xml": str(xml_path),
        "title": title,
        "doi": doi,
        "url": url,
        "year": year,
        "journal": journal,
        "publisher": publisher,
        "authors": parse_authors(root),
    }

def section_tags(head_text: str | None, n_attr: str | None, content: str) -> set[str]:
    head = (head_text or "").lower()
    n = (n_attr or "").strip()
    tags = set()

    if "summary" in head or "abstract" in head:
        tags.add("abstract")

    if "intro" in head or n == "1" or n.startswith("1."):
        tags.add("introduction")

    if any(k in head for k in [
        "study area",
        "study region",
        "study site",
        "area of interest",
        "study area and data",
        "study region and data",
    ]):
        tags.add("study_area")

    if any(k in head for k in [
        "dataset",
        "datasets",
        "data sources",
        "data",
        "materials",
    ]):
        tags.add("data_sources")

    if any(k in head for k in [
        "method",
        "methods",
        "methodology",
        "workflow",
        "processing",
        "model",
        "models",
        "simulation",
        "algorithm",
        "algorithms",
        "change detection algorithms",
    ]):
        tags.add("methods")

    if "data and methods" in head or "materials and methods" in head:
        tags.update({"methods", "data_sources"})

    if any(k in head for k in [
        "result",
        "results",
        "accuracy",
        "evaluation",
        "assessment",
    ]):
        tags.add("results")

    if "discussion" in head:
        tags.add("discussion")

    if "conclusion" in head:
        tags.add("conclusion")

    if not tags:
        tags.add("other")

    return tags

def split_inline_sections(content: str) -> dict:
    content = content or ""
    found = []

    for name, pattern in INLINE_HEADINGS:
        for m in re.finditer(pattern, content, flags=re.I):
            found.append((m.start(), m.end(), name))

    if not found:
        return {"other": content}

    found.sort()
    result = {}

    prefix = clean_text(content[:found[0][0]])
    if prefix:
        result.setdefault("other", "")
        result["other"] += prefix + "\n"

    for i, (start, end, name) in enumerate(found):
        next_start = found[i + 1][0] if i + 1 < len(found) else len(content)
        block = clean_text(content[end:next_start])

        if block:
            result.setdefault(name, "")
            result[name] += block + "\n"

    return result

def parse_sections(root) -> dict:
    sections = {
        "abstract": "",
        "introduction": "",
        "study_area": "",
        "data_sources": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "conclusion": "",
        "other": "",
    }

    abstract = root.xpath("//tei:abstract//tei:p//text()", namespaces=NS)
    if abstract:
        sections["abstract"] += clean_text(" ".join(abstract)) + "\n"

    for div in root.xpath("//tei:body//tei:div", namespaces=NS):
        head = all_text(div, "./tei:head//text()")
        n_attr = div.get("n")
        content = clean_text(" ".join(div.xpath(".//tei:p//text()", namespaces=NS)))

        if not content:
            continue

        tags = section_tags(head, n_attr, content)

        if tags == {"other"}:
            inline_sections = split_inline_sections(content)

            for tag, block in inline_sections.items():
                if tag in sections:
                    sections[tag] += block + "\n"

        else:
            for tag in tags:
                if tag in sections:
                    sections[tag] += content + "\n"

    return sections

def extract_tei_countries(root) -> list[dict]:
    found = {}

    for c in root.xpath("//tei:country", namespaces=NS):
        name = normalize_country_name(
            clean_text(" ".join(c.xpath(".//text()")))
        )
        code = c.get("key")

        if name:
            found[name] = {
                "name": name,
                "code": code,
                "source": "tei",
                "confidence": 0.95,
            }

    return list(found.values())

def normalize_country_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return clean_text(name).replace(";", "").strip()

def ensure_country_dict(c):
    if isinstance(c, dict):
        name = normalize_country_name(c.get("name"))
        return {
            **c,
            "name": name,
            "source": c.get("source", "unknown"),
            "confidence": c.get("confidence", 0.5),
        }

    return {
        "name": normalize_country_name(c),
        "code": None,
        "source": "unknown",
        "confidence": 0.5,
    }

def ensure_list_of_country_dicts(items):
    if not items:
        return []

    result = []
    seen = set()

    for item in items:
        c = ensure_country_dict(item)
        name = c.get("name")

        if not name or name in seen:
            continue

        seen.add(name)
        result.append(c)

    return result

def ensure_river_dict(r):
    if isinstance(r, dict):
        name = clean_text(r.get("name"))
        return {
            **r,
            "name": name,
            "type": r.get("type", "river"),
            "source": r.get("source", "unknown"),
        }

    return {
        "name": clean_text(str(r)),
        "type": "river",
        "source": "unknown",
    }

def ensure_list_of_river_dicts(items):
    if not items:
        return []

    result = []
    seen = set()

    for item in items:
        r = ensure_river_dict(item)
        name = r.get("name")

        if not name or name in seen:
            continue

        seen.add(name)
        result.append(r)

    return result

def detect_countries(text: str, root) -> list[dict]:
    found = {}

    for c in extract_tei_countries(root):
        c = ensure_country_dict(c)
        found[c["name"]] = c

    for name, pattern in COUNTRY_PATTERNS.items():
        m = re.search(pattern, text, re.I)
        if m and name not in found:
            found[name] = {
                "name": name,
                "code": None,
                "source": "regex",
                "confidence": 0.75,
                "evidence": snippet(text, m.start(), m.end()),
            }

    return ensure_list_of_country_dicts(list(found.values()))

def detect_study_countries_from_text(text: str) -> list[dict]:
    found = {}

    for name, pattern in COUNTRY_PATTERNS.items():
        m = re.search(pattern, text, re.I)
        if m and name not in found:
            found[name] = {
                "name": name,
                "code": None,
                "source": "regex_study_text",
                "confidence": 0.75,
                "evidence": snippet(text, m.start(), m.end()),
            }

    return ensure_list_of_country_dicts(list(found.values()))

def normalize_river_name(name: str) -> str:
    name = clean_text(name)
    name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.I)
    name = re.sub(r"'s$", "", name)
    name = name.replace(" River basin", "").replace(" river basin", "")
    name = name.replace(" River", "").replace(" river", "")
    name = clean_text(name)

    if name.lower() == "cagayan":
        return "Cagayan River"

    return name

def extract_rivers(text: str):
    text = text or ""
    entities = {}

    for pattern in RIVER_PATTERNS:
        for m in re.finditer(pattern, text):
            raw = clean_text(m.group(1))
            name = normalize_river_name(raw)

            if len(name) < 3:
                continue

            if name not in entities:
                ent = make_entity(
                    name=name,
                    etype="river",
                    evidence=snippet(text, m.start(), m.end()),
                    source="regex"
                )
                ent["scores"]["pattern"] = 0.8
                entities[name] = ent

    return list(entities.values())

def enrich_river_country(rivers: list[dict]) -> list[dict]:
    links = []

    rivers = ensure_list_of_river_dicts(rivers)

    for river in rivers:
        name = river.get("name")

        if name in RIVER_TO_COUNTRY:
            links.append({
                "river": name,
                "country": RIVER_TO_COUNTRY[name],
                "source": "lookup",
                "confidence": 0.85,
            })

    return links

def refine_study_area_with_embeddings(name: str, text: str):
    if not name:
        return name

    emb_name = model.encode([name])[0]
    emb_text = model.encode([text])[0]

    sim = cosine_similarity([emb_name], [emb_text])[0][0]

    if sim < 0.2:
        return None  # шум

    return name

def is_valid_study_type_label(label: str | None) -> bool:
    return label in VALID_STUDY_TYPES

def is_study_area_context(ctx: str) -> bool:
    ctx = (ctx or "").lower()

    positive = any(k in ctx for k in [
        "study area",
        "study site",
        "test site",
        "tested on",
        "tested using",
        "we tested",
        "we applied",
        "case study",
        "flood event",
        "flood events",
        "occurred in",
        "located in",
        "ground truth data",
        "validation data",
        "data sources",
        "two urban flood events",
    ])

    negative = any(k in ctx for k in [
        "previous studies",
        "several studies",
        "many studies",
        "literature",
        "ref.",
        "et al.",
        "state of the art",
        "existing methods",
        "in contrast",
        "for example",
    ])

    return positive and not negative

def sanitize_study_type(study_type_obj: dict) -> dict:
    study_type_obj = ensure_dict(study_type_obj)

    label = study_type_obj.get("label")

    if label not in VALID_STUDY_TYPES:
        return {
            "label": "unknown",
            "confidence": 0.4,
            "source": "sanitized_invalid_study_type",
            "previous": label,
            "needs_judge": True,
        }

    return study_type_obj

def detect_study_type(sections: dict, title: str = "") -> dict:
    sections = ensure_dict(sections)

    title_text = title or ""
    abstract = sections.get("abstract", "")
    study_area = sections.get("study_area", "")
    methods = sections.get("methods", "")[:1500]
    results = sections.get("results", "")[:1000]

    title_abs = f"{title_text} {abstract}".lower()
    strong_text = f"{title_text} {abstract} {study_area} {methods} {results}".lower()

    flood_case_evidence = (
        "flood" in strong_text
        and any(k in strong_text for k in [
            "tested on",
            "tested using",
            "we tested",
            "we applied",
            "applied to",
            "validated on",
            "evaluated on",
            "study area",
            "study site",
            "case study",
            "case area",
            "test site",
            "pilot area",
            "event occurred",
            "occurred in",
            "flood event",
            "flood events",
            "flooded area",
            "flood extent",
            "ground truth data",
            "ground truth data collected",
            "post-flood",
            "preflood",
            "pre-flood",
        ])
    )

    general_case_evidence = any(k in strong_text for k in [
        "tested on",
        "tested using",
        "we tested",
        "we applied",
        "applied to",
        "validated on",
        "evaluated on",
        "study area",
        "study site",
        "case study",
        "case area",
        "test site",
        "pilot area",
        "event occurred",
        "occurred in",
        "study was conducted",
        "dataset was collected",
    ])

    regional_evidence = any(k in strong_text for k in [
        "regional scale",
        "national scale",
        "country scale",
        "across the country",
        "entire country",
        "whole country",
        "territory of",
        "large region",
        "administrative region",
        "natural zones",
        "multiple stations",
        "weather stations",
        "meteorological stations",
    ])

    multi_site_evidence = any(k in strong_text for k in [
        "multiple case studies",
        "several case studies",
        "five case studies",
        "multiple locations",
        "multiple sites",
        "different locations",
        "different regions",
        "various regions",
        "multiple flood events",
        "several flood events",
        "two flood events",
        "different test sites",
        "benchmark sites",
    ])

    global_evidence = any(k in strong_text for k in [
        "global scale",
        "global-scale",
        "worldwide",
        "anywhere in the world",
        "global application",
        "global flood monitoring",
        "global basis",
        "near real-time on a global basis",
        "globally available",
        "global datasets",
        "worldwide application",
    ])

    review_title_abs_evidence = any(k in title_abs for k in [
        "systematic review",
        "literature review",
        "review paper",
        "this review",
        "we review",
        "review of",
        "state-of-the-art review",
        "survey of",
        "overview of",
        "meta-analysis",
    ])

    weak_review_noise = any(k in strong_text for k in [
        "previous studies",
        "several studies",
        "many studies",
        "existing methods",
        "state of the art",
        "related work",
        "literature",
        "literature review",
    ])

    if review_title_abs_evidence and not (flood_case_evidence or general_case_evidence):
        return {
            "label": "review",
            "confidence": 0.95,
            "source": "rules_title_abstract",
            "needs_judge": False,
        }

    if multi_site_evidence:
        return {
            "label": "multi_site",
            "confidence": 0.88,
            "source": "rules",
            "needs_judge": False,
        }

    if flood_case_evidence:
        return {
            "label": "case_study",
            "confidence": 0.92,
            "source": "rules_flood_case",
            "needs_judge": False,
        }

    if regional_evidence and not general_case_evidence:
        return {
            "label": "regional",
            "confidence": 0.88,
            "source": "rules",
            "needs_judge": False,
        }

    if general_case_evidence:
        return {
            "label": "case_study",
            "confidence": 0.9,
            "source": "rules",
            "needs_judge": False,
        }

    if global_evidence:
        return {
            "label": "global_algorithmic",
            "confidence": 0.85,
            "source": "rules",
            "needs_judge": bool(weak_review_noise),
        }

    label, scores = classify_with_embeddings(strong_text)
    confidence = float(scores[label])

    if label not in VALID_STUDY_TYPES:
        return {
            "label": "unknown",
            "confidence": 0.4,
            "source": "embeddings_invalid_label",
            "needs_judge": True,
            "scores": scores,
        }

    return {
        "label": label,
        "confidence": confidence,
        "source": "embeddings",
        "needs_judge": confidence < 0.85 or weak_review_noise,
        "scores": scores,
    }

def extract_study_area_structured(text: str):
    result = {}

    # 🔥 назва (більш універсально)
    m = re.search(
        r"([A-Z][a-zA-Z\s\-]+(?:Reserve|Park|Basin|Catchment|Region))",
        text
    )
    if m:
        result["name"] = m.group(1).strip()

    # країна
    for country, pattern in COUNTRY_PATTERNS.items():
        if re.search(pattern, text, re.I):
            result["country"] = country
            break

    # регіон
    region_match = re.search(
        r"([A-Z][a-z]+ region|[A-Z][a-z]+ oblast)",
        text,
        re.I
    )
    if region_match:
        result["region"] = region_match.group(1)

    # координати (стабільніше)
    coord_match = re.search(
        r"(\d{2})[°\s]+(\d{2}).*?N.*?(\d{2})[°\s]+(\d{2}).*?E",
        text
    )

    if coord_match:
        lat1, lat2, lon1, lon2 = coord_match.groups()

        result["coordinates"] = {
            "lat_min": float(f"{lat1}.{lat2}"),
            "lat_max": float(f"{lat1}.{int(lat2)+5}"),
            "lon_min": float(f"{lon1}.{lon2}"),
            "lon_max": float(f"{lon1}.{int(lon2)+5}")
        }

    if result and result.get("name"):
        refined = refine_study_area_with_embeddings(
            result["name"],
            text
        )
        result["name"] = refined

    return result if result else None

def is_valid_region_name(name: str) -> bool:
    if not name:
        return False

    n = clean_text(name).lower()

    if len(n) < 3:
        return False

    if n in GENERIC_REGION_NOISE:
        return False

    return True

def region_context_score(region_name: str, text: str) -> float:
    text = text or ""
    region_name = region_name or ""

    matches = list(re.finditer(re.escape(region_name), text, re.I))

    if not matches:
        return 0.0

    query = f"actual study region or study area: {region_name}"
    emb_query = model.encode([query])[0]

    best = 0.0

    for m in matches[:5]:
        ctx = snippet(text, m.start(), m.end(), window=500)
        emb_ctx = model.encode([ctx])[0]
        sim = cosine_similarity([emb_query], [emb_ctx])[0][0]
        best = max(best, float(sim))

    return float(best)

def extract_regions(
    text: str,
    ner_locations: list[str] | None = None,
    invalid_names: set[str] | None = None
) -> list[dict]:
    text = text or ""
    invalid_names = invalid_names or set()
    candidates = {}

    for name, pattern in REGION_PATTERNS.items():
        m = re.search(pattern, text, re.I)

        if m and is_valid_region_name(name):
            candidates[name] = {
                "name": name,
                "type": "region",
                "source": "regex_region",
                "confidence": 0.85,
                "evidence": snippet(text, m.start(), m.end()),
            }

    for loc in ner_locations or []:
        loc_name = clean_text(loc)

        if not loc_name:
            continue

        if loc_name.lower() in invalid_names:
            continue

        if not is_valid_place(loc_name):
            continue

        if not is_valid_region_name(loc_name):
            continue

        loc_type = classify_location(loc_name)

        if loc_type not in {"region", "basin", "watershed"}:
            continue

        ctx = best_mention_context(loc_name, text, window=500)

        if not is_study_area_context(ctx):
            continue

        candidates.setdefault(loc_name, {
            "name": loc_name,
            "type": loc_type,
            "source": "ner_region_context",
            "confidence": 0.7,
            "evidence": ctx,
        })

    results = []

    for item in candidates.values():
        score = region_context_score(item["name"], text)
        item["semantic_score"] = round(float(score), 4)

        if item["source"] == "regex_region":
            item["confidence"] = min(float(item["confidence"]) + 0.05, 0.95)
            results.append(item)

        elif score >= 0.35:
            item["confidence"] = min(float(item["confidence"]) + float(score), 0.9)
            results.append(item)

    return sorted(results, key=lambda x: x["confidence"], reverse=True)

def normalize_regions(regions):
    if not regions:
        return []

    result = []

    for r in regions:
        if isinstance(r, dict):
            name = clean_text(r.get("name"))
            item = {
                **r,
                "name": name,
                "type": r.get("type", "region"),
                "source": r.get("source", "unknown"),
                "confidence": float(r.get("confidence", 0.5)),
            }
        else:
            name = clean_text(str(r))
            item = {
                "name": name,
                "type": "region",
                "source": "unknown",
                "confidence": 0.5,
            }

        if not is_valid_region_name(name):
            continue

        result.append(item)

    return result

def extract_dems(ctx: PipelineContext):
    return extract_pattern_entities_strict(
        ctx.satellite_text,
        DEM_DATASETS,
        "dem",
        "data_sources+methods"
    )

def extract_geo(root, ctx: PipelineContext, metadata: dict) -> dict:
    text = ctx.full_text

    # =========================
    # 🔥 study type
    # =========================
    study_type_obj = sanitize_study_type(
        detect_study_type(ctx.sections, metadata.get("title", ""))
    )

    # =========================
    # 🔥 author geo
    # =========================
    author_geo = ensure_list_of_country_dicts(
        extract_author_geo(metadata)
    )

    data_geo = extract_data_geo(ctx.sections)

    # =========================
    # 🔥 author blacklist
    # =========================
    authors = author_name_set(metadata)

    # =========================
    # 🔥 NER
    # =========================
    ner_countries, ner_locations = extract_geo_ner(text)

    # =========================
    # 🔥 study-country text window (ВАЖЛИВО)
    # =========================
    detected_countries = ensure_list_of_country_dicts(
        detect_study_countries_from_text(ctx.study_country_text)
    )

    # =========================
    # 🔥 structured study area
    # =========================
    study_area_struct = extract_study_area_structured(ctx.study_area)

    # =========================
    # 🔥 rivers
    # =========================
    rivers = ensure_list_of_river_dicts(
        extract_rivers(text)
    )

    river_links = enrich_river_country(rivers)

    # =========================
    # 🔥 regions (з фільтром авторів)
    # =========================
    regions = normalize_regions(
        extract_regions(
            text,
            ner_locations,
            invalid_names=authors
        )
    )

    # =========================
    # 🔥 locations (context-aware)
    # =========================
    locations = []

    for loc in ner_locations:
        loc_name = clean_text(loc)

        if not loc_name:
            continue

        if loc_name.lower() in authors:
            continue

        if not is_valid_place(loc_name):
            continue

        ctx_snippet = best_mention_context(loc_name, text, window=500)

        if not is_study_area_context(ctx_snippet):
            continue

        locations.append({
            "name": loc_name,
            "type": classify_location(loc_name),
            "source": "ner_study_area_context",
            "confidence": 0.75,
            "evidence": ctx_snippet,
        })

    # =========================
    # 🔥 primary country
    # =========================
    primary_country = None

    if study_area_struct and study_area_struct.get("country"):
        primary_country = study_area_struct["country"]

    elif detected_countries:
        primary_country = detected_countries[0]["name"]

    elif river_links:
        primary_country = river_links[0]["country"]

    # =========================
    # 🔥 build study_geo
    # =========================
    study_geo = default_study_geo()
    study_geo.update({
        "primary_country": primary_country,
        "countries": detected_countries,
        "regions": regions,
        "rivers": rivers,
        "river_country_links": river_links,
        "locations": locations,
        "coordinates": [],
        "confidence": 0.0,
    })

    # =========================
    # 🔥 NER validation (НЕ ЛАМАЄ СТРУКТУРУ)
    # =========================
    study_geo = validate_with_ner(
        study_geo,
        ner_countries,
        ner_locations
    )

    # =========================
    # 🔥 RETURN WRAPPER (ВАЖЛИВО)
    # =========================
    return {
        "study_type": study_type_obj,
        "author_geo": author_geo,
        "study_geo": study_geo,
        "data_geo": data_geo,
        "note": "context-aware geo extraction"
    }

def geocode_place(name: str):
    if not is_valid_place(name):
        return None

    if name in GEO_CACHE:
        return GEO_CACHE[name]

    # 🔥 1. СПОЧАТКУ GeoNames
    try:
        g = geonames_lookup(name)

        if g:
            res = {
                "lat": g["lat"],
                "lon": g["lon"],
                "country": g.get("country"),
                "feature": g.get("feature"),
                "source": "geonames"
            }

            GEO_CACHE[name] = res
            return res

    except Exception:
        pass

    # 🔥 2. FALLBACK → Nominatim
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": name,
            "format": "json",
            "limit": 1
        }

        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": "geo-parser"}
        )
        data = r.json()

        if data:
            res = {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "source": "nominatim"
            }

            GEO_CACHE[name] = res
            return res

    except Exception:
        pass

    GEO_CACHE[name] = None
    return None

def enrich_with_coordinates(geo: dict) -> dict:
    enriched = []
    seen = set()

    countries = ensure_list_of_country_dicts(geo.get("countries", []))
    rivers = ensure_list_of_river_dicts(geo.get("rivers", []))

    geo["countries"] = countries
    geo["rivers"] = rivers

    for c in countries:
        name = c.get("name")

        if not is_valid_place(name) or name in seen:
            continue

        seen.add(name)
        res = geocode_place(name)

        if res:
            enriched.append({
                "name": name,
                "type": "country",
                "lat": res["lat"],
                "lon": res["lon"],
                "source": res.get("source")
            })

    for loc in geo.get("locations", []):
        if not isinstance(loc, dict):
            continue

        name = loc.get("name")
        conf = float(loc.get("confidence", 0.0))

        if conf < 0.7:
            continue

        if loc.get("source") == "ner" and not loc.get("evidence"):
            continue

        if not is_valid_place(name) or name in seen:
            continue

        seen.add(name)
        res = geocode_place(name)

        if res:
            enriched.append({
                "name": name,
                "type": loc.get("type"),
                "lat": res["lat"],
                "lon": res["lon"],
                "source": res.get("source")
            })

    for r in rivers:
        name = r.get("name")

        if not is_valid_place(name) or name in seen:
            continue

        seen.add(name)
        res = geocode_place(name)

        if res:
            enriched.append({
                "name": name,
                "type": "river",
                "lat": res["lat"],
                "lon": res["lon"],
                "source": res.get("source")
            })

    geo["coordinates"] = enriched
    return geo

def extract_pattern_entities(text: str, patterns: dict, field: str, section: str) -> list[dict]:
    results = []
    seen = set()

    for name, pattern in patterns.items():
        match = re.search(pattern, text, re.I)
        if match and name not in seen:
            seen.add(name)
            results.append({
                "name": name,
                "evidence": {
                    "field": field,
                    "section": section,
                    "snippet": snippet(text, match.start(), match.end()),
                    "source": "regex",
                }
            })

    return results

def normalize_metric_value(value: str, metric_type: str) -> float:
    v = float(value)

    if metric_type in {"OA", "F1", "IoU", "Kappa", "R", "Percent"} and v > 1:
        return round(v / 100, 4)

    return round(v, 4)

def extract_metrics(text: str) -> list[dict]:
    metrics = []
    seen = set()

    for metric_type, pattern in METRIC_PATTERNS.items():
        for m in re.finditer(pattern, text, re.I):

            if metric_type in {"RMSE", "MAE", "R", "p_value"}:
                raw = m.group(1)
            else:
                raw = m.group(2)

            key = (metric_type, raw)

            if key in seen:
                continue

            seen.add(key)

            value = normalize_metric_value(raw, metric_type)

            metrics.append({
                "type": metric_type,
                "value": value,
                "evidence": {
                    "field": metric_type,
                    "section": "results",
                    "snippet": snippet(text, m.start(), m.end()),
                    "source": "regex",
                }
            })

    return metrics

def extract_methods(ctx: PipelineContext):
    return extract_pattern_entities_strict(
        ctx.method_text,
        METHOD_PATTERNS,
        "method",
        "methods+results"
    )

def extract_data_geo(sections: dict) -> dict:
    text = " ".join([
        sections.get("abstract", ""),
        sections.get("introduction", ""),
        sections.get("methods", ""),
        sections.get("results", ""),
    ]).lower()

    flags = []

    if "global coverage" in text or "earth's entire surface" in text or "global scale" in text:
        flags.append("global")

    if "multiple flood images" in text or "hundreds of" in text or "eight data sets" in text:
        flags.append("multi_site")

    if "modis archive" in text or "earth engine" in text:
        flags.append("global_satellite_archive")

    return {
        "scope": flags or ["unknown"],
        "source": "text_semantic_rules",
        "confidence": 0.85 if flags else 0.4,
    }

def extract_satellites(ctx: PipelineContext):
    return extract_pattern_entities_strict(
        ctx.satellite_text,
        SATELLITE_PATTERNS,
        "satellite",
        "data_sources+methods"
    )

def make_entity(name, etype, evidence="", source="unknown"):
    return {
        "name": name,
        "type": etype,
        "evidence": evidence,
        "sources": [source],
        "scores": {
            "pattern": 0.0,
            "context": 0.0,
            "embedding": 0.0,
            "llm": 0.0
        },
        "final_score": 0.0,
        "accepted": None,
        "role": None  # used | mentioned
    }

def is_real_usage(ctx: str) -> bool:
    ctx = ctx.lower()

    positive = any(k in ctx for k in [
        "used",
        "applied",
        "we used",
        "this study uses",
        "data used",
        "method used",
        "we applied",
    ])

    negative = any(k in ctx for k in [
        "previous studies",
        "other studies",
        "review",
        "for example",
        "such as",
        "e.g.",
        "et al",
    ])

    return positive and not negative

def score_entity(entity, ctx_text):
    score = 0.0

    # 1. pattern
    if entity.get("source") == "regex":
        score += 0.3

    # 2. context
    if is_real_usage(entity.get("evidence", "")):
        score += 0.3

    # 3. embedding relevance
    emb_entity = model.encode([entity["name"]])[0]
    emb_text = model.encode([ctx_text])[0]

    sim = cosine_similarity([emb_entity], [emb_text])[0][0]
    score += sim * 0.2

    # 4. LLM (optional)
    score += entity.get("llm_score", 0.0) * 0.2

    return round(score, 3)

def compute_entity_score(entity):
    s = entity["scores"]

    score = (
        s.get("pattern", 0) * 0.3 +
        s.get("context", 0) * 0.3 +
        s.get("embedding", 0) * 0.2 +
        s.get("llm", 0) * 0.2
    )

    entity["final_score"] = round(score, 4)
    return entity

def decide_entity(entity, threshold=0.6):
    entity = compute_entity_score(entity)

    entity["accepted"] = entity["final_score"] >= threshold
    return entity

def filter_entities(entities, threshold=0.6):
    return [
        e for e in entities
        if score_entity(e, e.get("evidence", "")) >= threshold
    ]

def resolve_geo_entity(entity, context_text):
    geo = geonames_lookup(entity["name"])

    if not geo:
        return entity

    score = 0

    if geo["country"] and geo["country"].lower() in context_text.lower():
        score += 0.5

    if geo["type"] == entity["type"]:
        score += 0.3

    entity["geo"] = geo
    entity["scores"]["context"] += score

    return entity

def extract_pattern_entities_strict(
    text: str,
    patterns: dict,
    entity_type: str,
    source: str
):
    results = {}

    for name, pattern in patterns.items():
        for m in re.finditer(pattern, text, re.I):

            ctx = snippet(text, m.start(), m.end(), window=200)

            if not is_real_usage(ctx):
                continue

            key = name.lower()

            if key not in results:
                results[key] = {
                    "name": name,
                    "type": entity_type,
                    "source": source,
                    "confidence": 0.9,
                    "evidence": ctx
                }

    return list(results.values())


def extract_entities(root, ctx: PipelineContext, metadata):
    geo = extract_geo(root, ctx, metadata)

    satellites = extract_satellites(ctx)
    dems = extract_dems(ctx)
    methods = extract_methods(ctx)

    metrics = extract_metrics(ctx.metric_text)

    return {
        "geo": geo,
        "satellites": satellites,
        "dems": dems,
        "methods": methods,
        "metrics": metrics,
    }

def process_entities(entities, ctx: PipelineContext, judge=None):
    results = []

    for e in entities:
        e = add_context_score(e, ctx.full_text)
        e = detect_role(e, ctx.full_text)
        e = resolve_geo_entity(e, ctx.full_text)

        e = compute_entity_score(e)

        if judge:
            try:
                jr = judge.judge({"entity": e}, ctx.sections)
                e = apply_llm_judge(e, jr)
            except Exception:
                pass

        e = decide_entity(e)

        if e["accepted"]:
            results.append(e)

    return results
def run_entity_pipeline(entities, ctx, judge=None):
    result = []

    for e in entities:
        e = add_context_score(e, ctx.full_text)
        e = detect_role(e, ctx.full_text)

        # 👉 тут твій embedding (якщо треба)
        # e["scores"]["embedding"] = ...

        # 👉 decision
        score = (
            e["scores"].get("pattern", 0) * 0.5 +
            e["scores"].get("context", 0) * 0.5
        )

        e["final_score"] = score
        e["accepted"] = score > 0.6

        # 👉 LLM judge (опціонально)
        if judge:
            try:
                jr = judge.judge({"entity": e}, ctx.sections)
                if not jr.get("accepted", True):
                    e["accepted"] = False
            except:
                pass

        if e["accepted"]:
            result.append(e)

    return result
def infer_sensor_types(satellites: list[dict], dems: list[dict]) -> list[str]:
    sar = {"Sentinel-1", "RADARSAT", "TerraSAR-X", "COSMO-SkyMed", "ALOS PALSAR"}
    optical = {"Sentinel-2", "Landsat", "MODIS", "VIIRS", "PlanetScope", "WorldView", "Pleiades"}
    lidar = {"ICESat-2"}

    found = set()

    for sat in satellites:
        name = sat["name"]
        if name in sar:
            found.add("SAR")
        elif name in optical:
            found.add("Optical")
        elif name in lidar:
            found.add("LiDAR")
    if dems:
        found.add("DEM")

    return sorted(found)

def validate_task(label, text):
    t = text.lower()

    if label == "flood_mapping_satellite":
        if not any(k in t for k in ["satellite", "sar", "sentinel", "landsat"]):
            return False

    if label == "flood_modeling_hydraulic":
        if not any(k in t for k in ["hec-ras", "hydraulic", "2d"]):
            return False

    if label == "hydrological_modeling":
        if not any(k in t for k in ["swat", "hec-hms", "runoff"]):
            return False

    return True

def classify_task(ctx: PipelineContext):
    text = ctx.full_text.lower()

    if "flood" in text and any(k in text for k in [
        "mapping", "extent", "inundation", "water extent"
    ]):
        return make_task("flood_mapping_satellite", 0.9)

    if any(k in text for k in [
        "ndvi", "ndwi", "ndii", "rdi"
    ]):
        return make_task("spectral_index_analysis", 0.88)

    if any(k in text for k in [
        "hec-ras", "hydraulic", "2d flood"
    ]):
        return make_task("flood_modeling_hydraulic", 0.88)

    if any(k in text for k in [
        "swat", "hec-hms", "runoff"
    ]):
        return make_task("hydrological_modeling", 0.88)

    label, scores = classify_task_with_embeddings(text)

    if label not in VALID_TASK_LABELS:
        return make_task("unknown", 0.4)

    return make_task(label, scores[label], source="embeddings")

def needs_judge(paper: dict) -> bool:
    entities = ensure_dict(paper.get("entities"))
    geo = ensure_dict(entities.get("geo"))
    study_geo = ensure_dict(geo.get("study_geo"))
    study_type = ensure_dict(geo.get("study_type"))
    task = ensure_dict(entities.get("task"))

    if study_type.get("needs_judge") is True:
        return True

    if study_type.get("confidence", 1.0) < 0.85:
        return True

    if study_type.get("label") in {"global_algorithmic", "multi_site"}:
        if study_geo.get("primary_country") or study_geo.get("rivers"):
            return True

    if study_geo.get("confidence", 1.0) < 0.75:
        return True

    if study_geo.get("rivers") and study_geo.get("confidence", 1.0) < 0.9:
        return True

    if entities.get("dems"):
        return True

    if entities.get("satellites") and entities.get("dems"):
        return True

    if task.get("confidence", 1.0) < 0.85:
        return True

    return False

def is_valid_judge_task_verdict(task_verdict: dict) -> bool:
    task_verdict = ensure_dict(task_verdict)

    accepted = task_verdict.get("accepted")
    original = task_verdict.get("original_value")
    corrected = task_verdict.get("corrected_value")

    if corrected and corrected not in VALID_TASK_LABELS:
        return False

    if accepted is True and corrected not in {None, original}:
        return False

    return True

def is_valid_judge_study_type_verdict(verdict: dict) -> bool:
    verdict = ensure_dict(verdict)

    accepted = verdict.get("accepted")
    original = verdict.get("original_value")
    corrected = verdict.get("corrected_value")

    if original and original not in VALID_STUDY_TYPES:
        return False

    if corrected and corrected not in VALID_STUDY_TYPES:
        return False

    if accepted is True and corrected not in {None, original}:
        return False

    return True

def is_ollama_available(base_url="http://localhost:11434") -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def judge_paper_with_ollama(paper: dict) -> dict:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "mistral-nemo:12b")

    if not is_ollama_available(base_url):
        return {
            "status": "skipped",
            "reason": f"Ollama is not available at {base_url}",
        }

    sections = ensure_dict(paper.get("sections"))
    metadata = ensure_dict(paper.get("metadata"))
    entities = ensure_dict(paper.get("entities"))

    candidate = {
        "paper_id": metadata.get("paper_id"),
        "title": metadata.get("title"),

        "study_type": ensure_dict(
            ensure_dict(entities.get("geo")).get("study_type")
        ),

        "study_geo": ensure_dict(
            ensure_dict(entities.get("geo")).get("study_geo")
        ),

        "task": ensure_dict(entities.get("task")),

        "author_geo": ensure_dict(entities.get("geo")).get("author_geo", []),
        "satellites": entities.get("satellites", []),
        "dems": entities.get("dems", []),
        "methods": entities.get("methods", []),
    }

    judge = OllamaJudge(
        base_url=base_url,
        model=model_name,
        timeout=180,
    )

    return judge.judge(candidate, sections)

def apply_judge_verdict(paper: dict, verdict: dict) -> dict:
    verdict = ensure_dict(verdict)

    if verdict.get("status") in {"skipped", "failed"}:
        return paper

    entities = paper.setdefault("entities", {})
    geo = entities.setdefault("geo", {})

    geo["study_geo"] = ensure_dict(
        geo.get("study_geo"),
        default_study_geo()
    )

    study_geo = geo["study_geo"]

    entities.setdefault("validation_warnings", [])

    study_type_verdict = ensure_dict(verdict.get("study_type"))

    if study_type_verdict:
        if not is_valid_judge_study_type_verdict(study_type_verdict):
            entities["validation_warnings"].append({
                "type": "invalid_llm_study_type_verdict",
                "verdict": study_type_verdict,
            })
        elif not study_type_verdict.get("accepted", True):
            corrected = study_type_verdict.get("corrected_value")
            conf = float(study_type_verdict.get("confidence", 0.7))

            if corrected and conf >= 0.8:
                geo["study_type"] = {
                    "label": corrected,
                    "confidence": conf,
                    "source": "ollama_judge",
                    "previous": study_type_verdict.get("original_value"),
                    "reason": study_type_verdict.get("reason"),
                }

    country_verdict = ensure_dict(verdict.get("study_country"))

    if country_verdict:
        accepted = country_verdict.get("accepted")
        original = country_verdict.get("original_value")
        corrected = country_verdict.get("corrected_value")
        conf = float(country_verdict.get("confidence", 0.7))

        final_country = corrected if corrected else original

        if final_country and conf >= 0.8 and not study_geo.get("primary_country"):
            study_geo["primary_country"] = final_country
            study_geo["countries"] = [{
                "name": final_country,
                "source": "ollama_judge",
                "confidence": conf,
                "reason": country_verdict.get("reason"),
            }]

    task_verdict = ensure_dict(verdict.get("task"))

    if task_verdict:
        if not is_valid_judge_task_verdict(task_verdict):
            entities["validation_warnings"].append({
                "type": "invalid_llm_task_verdict",
                "verdict": task_verdict,
            })
        elif not task_verdict.get("accepted", True):
            corrected = task_verdict.get("corrected_value")
            conf = float(task_verdict.get("confidence", 0.7))

            if corrected and conf >= 0.8:
                entities["task"] = {
                    "label": corrected,
                    "confidence": conf,
                    "source": "ollama_judge",
                    "previous": task_verdict.get("original_value"),
                    "reason": task_verdict.get("reason"),
                }

    return paper

def apply_llm_judge(entity, judge_result):
    if not judge_result:
        return entity

    accepted = judge_result.get("accepted", True)
    confidence = judge_result.get("confidence", 0.5)

    entity["scores"]["llm"] = confidence

    if not accepted:
        entity["final_score"] *= 0.5
        entity["accepted"] = False

    return entity

def build_paper_json(xml_path: Path) -> dict:
    # =========================
    # 🔥 parse XML
    # =========================
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    metadata = parse_metadata(root, xml_path)
    sections = parse_sections(root)

    # =========================
    # 🔥 ctx (ПІСЛЯ sections!)
    # =========================
    ctx = PipelineContext(sections)

    # =========================
    # 🔥 entities (вже з ctx)
    # =========================
    entities = extract_entities(root, ctx, metadata)

    # =========================
    # 🔥 sensor types
    # =========================
    entities["sensor_types"] = infer_sensor_types(
        entities.get("satellites", []),
        entities.get("dems", []),
    )

    # =========================
    # 🔥 task
    # =========================
    entities["task"] = classify_task(ctx)

    # =========================
    # 🔥 metadata
    # =========================
    full_text = ctx.full_text

    paper = {
        "metadata": {
            **metadata,
            "content_hash": make_hash(full_text),
        },
        "sections": sections,
        "entities": entities,
        "llm_judge": None,
        "provenance": {
            "parser": "grobid_tei_ctx_v1",
            "source_xml": str(xml_path),
            "has_geo": bool(
                entities.get("geo", {})
                .get("study_geo", {})
                .get("primary_country")
            ),
            "judge_used": False,
            "judge_model": None,
        }
    }

    return json_safe(paper)

def run():
    xml_files = sorted(XML_DIR.glob("*.tei.xml"))

    print(f"🔍 Found {len(xml_files)} XML files")

    for xml_file in xml_files:
        try:
            print(f"📄 Building JSON: {xml_file.name}")

            paper = build_paper_json(xml_file)

            out_path = OUT_DIR / f"{xml_file.stem}.paper.json"

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(json_safe(paper), f, indent=2, ensure_ascii=False)

            print(f"✅ Saved: {out_path.name}")

        except Exception as e:
            print(f"❌ Error in {xml_file.name}: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    run()