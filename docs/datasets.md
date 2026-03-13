# ForestVision Datasets

This document provides comprehensive documentation for the `forestvision.datasets` module, which provides dataset classes for geospatial and remote sensing data with a focus on forest-related datasets.

## Table of Contents

- [Overview](#overview)
- [Class Hierarchy](#class-hierarchy)
- [Abstract Base Classes](#abstract-base-classes)
  - [CloudRasterDataset](#cloudrasterdataset)
  - [GEERasterDataset](#geerasterdataset)
- [Google Earth Engine Datasets](#google-earth-engine-datasets)
  - [GEESentinel2](#geesentinel2)
  - [GEELandsat8](#geelandsat8)
  - [GEELandsatTimeSeries](#geelandsattimeseries)
  - [GEELandsatFTV](#geelandsatftv)
  - [GEELandTrendr](#geelandtrendr)
  - [GEELandTrendrDisturbance](#geelandtrendrdisturbance)
  - [GEEDynamicWorld](#geedynamicworld)
  - [GEEAlphaEarth](#geealphaearth)
  - [GEE3Dep](#gee3dep)
  - [GEEGlobalForestChange](#geeglobalforestchange)
- [Local Raster Datasets](#local-raster-datasets)
  - [eMapRAGB](#emapragb)
  - [ForestOwnership](#forestownership)
  - [GNNForestAttr](#gnnforestattr)
- [Utility Classes](#utility-classes)
  - [GPDFeatureCollection](#gpdfeaturecollection)
  - [DatasetStats](#datasetstats)
- [Helper Functions](#helper-functions)

---

## Overview

The `forestvision.datasets` module extends [TorchGeo](https://github.com/microsoft/torchgeo) datasets to provide seamless integration with:

- **Google Earth Engine (GEE)** - Cloud-based satellite imagery with automatic downloading and caching
- **Local raster files** - GeoTIFF and other raster formats for ground truth data
- **Vector data** - GeoDataFrame integration for spatial sampling

### Key Features

- **Unified interface** - All datasets follow the TorchGeo `GeoDataset` pattern
- **Automatic caching** - Downloaded GEE data is cached locally
- **Queue management** - Thread-safe request queue for GEE API calls
- **Metadata tracking** - STAC-like metadata for dataset provenance
- **Visualization** - Built-in plotting methods for quick data inspection

---

## Class Hierarchy

```
GeoDataset (torchgeo)
    └── CloudRasterDataset (abstract)
            └── GEERasterDataset (abstract)
                    ├── GEESentinel2
                    ├── GEELandsat8
                    ├── GEELandsatFTV
                    ├── GEELandTrendrDisturbance
                    ├── GEEDynamicWorld
                    ├── GEEAlphaEarth
                    ├── GEE3Dep
                    └── GEEGlobalForestChange

RasterDataset (torchgeo)
    ├── eMapRAGB
    ├── ForestOwnership
    └── GNNForestAttr

GEELandsatTimeSeries (callable class)
GEELandTrendr (analysis class)
```

---

## Abstract Base Classes

### CloudRasterDataset

**Source:** `forestvision/datasets/cloudgeo.py`

Abstract base class for imagery served from cloud data providers. Provides an interface for fetching geospatial imagery from cloud services like Google Earth Engine without requiring local data storage.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `is_image` | `bool` | Whether dataset contains image (True) or mask (False) data |
| `all_bands` | `List[str]` | List of all available bands |
| `rgb_bands` | `List[str]` | Bands to use for RGB visualization |
| `cmap` | `dict[int, Tuple]` | Color map for visualization |
| `dtype` | `torch.dtype` | Data type (float32 for images, long for masks) |

#### Constructor

```python
CloudRasterDataset(
    roi: BoundingBox,
    path: Optional[str] = None,
    res: Optional[float] = None,
    transforms: Optional[Callable] = None,
    crs: Optional[CRS] = None,
    download: bool = False,
    cache: bool = True,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `roi` | `BoundingBox` | Region of interest to fetch data from |
| `path` | `str` | Directory for caching downloaded data |
| `res` | `float` | Target resolution in meters per pixel |
| `transforms` | `Callable` | Transform function applied to each sample |
| `crs` | `CRS` | Coordinate reference system |
| `download` | `bool` | Whether to download data to path |
| `cache` | `bool` | Whether to cache data in memory |

#### Abstract Methods

Subclasses must implement `_get_pixels(query: BoundingBox) -> numpy.ndarray` to define how data is fetched from the cloud provider.

---

### GEERasterDataset

**Source:** `forestvision/datasets/geebase.py`

Abstract class for fetching imagery from Google Earth Engine. Extends `CloudRasterDataset` with GEE-specific functionality including queue management, metadata tracking, and automatic URL refresh.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `gee_asset_id` | `str` | GEE Earth Engine asset ID |
| `instrument` | `str` | Name of sensor/instrument |
| `nodata` | `int` | NoData value for the dataset |
| `date_start` | `str` | Start date for data collection (YYYY-MM-DD) |
| `date_end` | `str` | End date for data collection (YYYY-MM-DD) |

#### Constructor

```python
GEERasterDataset(
    roi: Optional[BoundingBox] = None,
    path: Optional[str] = None,
    res: Union[int, None] = None,
    crs: Optional[CRS] = CRS.from_epsg(5070),
    transforms: Optional[Callable] = None,
    download: bool = False,
    bypass_errors: bool = True,
    overwrite: bool = False,
    cache: bool = True,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `roi` | `BoundingBox` | Region of interest |
| `path` | `str` | Directory for local caching |
| `res` | `int/float` | Pixel resolution in meters |
| `crs` | `CRS` | CRS for fetching images (default: EPSG:5070) |
| `transforms` | `Callable` | Transform function for samples |
| `download` | `bool` | Download data to path |
| `bypass_errors` | `bool` | Log errors instead of raising |
| `overwrite` | `bool` | Overwrite existing cached files |
| `cache` | `bool` | Cache data in memory |

#### Abstract Methods

| Method | Description |
|--------|-------------|
| `collection` | Property returning `ee.ImageCollection` with filters applied |
| `_reducer(collection)` | Reduce collection to single `ee.Image` |
| `_preprocess(image)` | Preprocess Earth Engine image |

#### Queue Management

GEERasterDataset uses a thread-safe queue system for GEE API requests:

```python
from forestvision.datasets.geebase import start_gee_queue, stop_gee_queue

# Start queue with custom concurrency
start_gee_queue(max_concurrent=10, rate_limit_delay=0.05)

# Use datasets...

# Stop queue when done
stop_gee_queue()
```

---

## Google Earth Engine Datasets

### GEESentinel2

**Source:** `forestvision/datasets/geesentinel.py`

Sentinel-2 Surface Reflectance Harmonized image collection from Google Earth Engine.

#### Dataset Information

| Property | Value |
|----------|-------|
| **GEE Asset** | `COPERNICUS/S2_SR_HARMONIZED` |
| **Resolution** | 10 meters |
| **Instrument** | Sentinel-2 MSI |
| **NoData** | 0 |

#### Available Bands

```python
all_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
rgb_bands = ["B4", "B3", "B2"]  # Red, Green, Blue
```

| Band | Description |
|------|-------------|
| B1 | Coastal aerosol (60m) |
| B2 | Blue (10m) |
| B3 | Green (10m) |
| B4 | Red (10m) |
| B5 | Vegetation Red Edge (20m) |
| B6 | Vegetation Red Edge (20m) |
| B7 | Vegetation Red Edge (20m) |
| B8 | NIR (10m) |
| B8A | Vegetation Red Edge (20m) |
| B9 | Water vapor (60m) |
| B11 | SWIR 1 (20m) |
| B12 | SWIR 2 (20m) |

#### Constructor

```python
GEESentinel2(
    year: Optional[int] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    bands: Optional[list[str]] = None,
    roi: Optional[BoundingBox] = None,
    season: str = "leafon",
    res: Optional[float] = None,
    path: Optional[str] = None,
    crs: Optional[CRS] = CRS.from_epsg(5070),
    transforms: Optional[Callable] = None,
    download: bool = False,
    overwrite: bool = False,
    cache: bool = True,
)
```

#### Usage Example

```python
from torchgeo.datasets import BoundingBox
from forestvision.datasets import GEESentinel2

# Define region of interest (minx, maxx, miny, maxy, mint, maxt)
roi = BoundingBox(-122.5, -122.4, 45.5, 45.6, 0, 1e12)

# Create dataset for summer 2023
dataset = GEESentinel2(
    year=2023,
    roi=roi,
    season="leafon",  # April-September
    bands=["B4", "B3", "B2", "B8"],  # RGB + NIR
    path="./data/sentinel2",
    download=True,
)

# Get a sample
sample = dataset[roi]
print(sample["image"].shape)  # [4, H, W] - 4 bands
```

#### Preprocessing

- Masks clouds and cirrus using QA60 band
- Filters images with cloud cover < 20%
- Creates median composite for the date range

---

### GEELandsat8

**Source:** `forestvision/datasets/geelandsat.py`

Landsat 8 Tier 1 Surface Reflectance data from Google Earth Engine with cloud masking and preprocessing.

#### Dataset Information

| Property | Value |
|----------|-------|
| **GEE Asset** | `LANDSAT/LC08/C02/T1_L2` |
| **Resolution** | 30 meters |
| **Instrument** | Landsat 8 OLI/TIRS |
| **NoData** | 0 |

#### Available Bands

```python
base_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
rgb_bands = ["SR_B6", "SR_B5", "SR_B4"]  # SWIR2, NIR, Red (false color)

# With spectral_index="TC", adds:
tc_bands = ["TCB", "TCG", "TCW", "TCA"]  # Tasseled Cap components
```

| Band | Description |
|------|-------------|
| SR_B1 | Coastal aerosol (30m) |
| SR_B2 | Blue (30m) |
| SR_B3 | Green (30m) |
| SR_B4 | Red (30m) |
| SR_B5 | NIR (30m) |
| SR_B6 | SWIR 1 (30m) |
| SR_B7 | SWIR 2 (30m) |
| TCB | Tasseled Cap Brightness |
| TCG | Tasseled Cap Greenness |
| TCW | Tasseled Cap Wetness |
| TCA | Tasseled Cap Angle |

#### Constructor

```python
GEELandsat8(
    year: int,
    roi: Optional[BoundingBox] = None,
    res: float = 30,
    season: str = "leafon",
    spectral_index: Optional[str] = None,
    spectral_index_only: bool = False,
    bands: Optional[List[str]] = None,
    path: Optional[str] = None,
    crs: Optional[CRS] = CRS.from_epsg(5070),
    transforms: Optional[Callable] = None,
    download: bool = False,
    overwrite: bool = False,
    cache: bool = True,
)
```

| Parameter | Description |
|-----------|-------------|
| `year` | Year of data |
| `season` | `"leafon"` (Apr-Sep) or `"leafoff"` (Oct-Mar) |
| `spectral_index` | `"TC"` for Tasseled Cap transform, or `None` |
| `spectral_index_only` | If True, return only TC bands |

#### Usage Example

```python
from forestvision.datasets import GEELandsat8

# Landsat with Tasseled Cap transform
dataset = GEELandsat8(
    year=2020,
    roi=roi,
    season="leafon",
    spectral_index="TC",
    spectral_index_only=False,  # Include both original and TC bands
)

sample = dataset[roi]
# sample["image"] contains [SR_B1-SR_B7, TCB, TCG, TCW, TCA]
```

---

### GEELandsatTimeSeries

**Source:** `forestvision/datasets/geelandsat.py`

Harmonized Landsat 5-8 time series imagery with medoid compositing. Creates consistent time series across Landsat sensors with cross-sensor harmonization.

#### Dataset Information

| Property | Value |
|----------|-------|
| **Sensors** | Landsat 5 TM, Landsat 7 ETM+, Landsat 8 OLI/TIRS |
| **Resolution** | 30 meters |
| **Time Range** | 1984 to present |

#### Harmonized Bands

```python
bands = ["B1", "B2", "B3", "B4", "B5", "B7"]
rgb_bands = ["B5", "B4", "B3"]  # SWIR1, NIR, Red
```

Band mapping harmonizes Landsat 8 bands to match Landsat 5/7:
- Landsat 8 SR_B2-4, SR_B5, SR_B6-7 -> Harmonized B1-5, B7

#### Constructor

```python
GEELandsatTimeSeries(
    roi: BoundingBox,
    date_start: int | str,
    date_end: int | str,
    season: str = "leafon",
    crs: Optional[CRS] = None,
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_tscollection()` | `ee.ImageCollection` | Time series with medoid compositing |
| `__call__()` | `ee.ImageCollection` | Same as `get_tscollection()` |

#### Usage Example

```python
from forestvision.datasets import GEELandsatTimeSeries

# Create 20-year time series
ts = GEELandsatTimeSeries(
    roi=roi,
    date_start=2000,
    date_end=2020,
    season="leafon",
)

# Get collection
collection = ts.get_tscollection()
# One medoid-composited image per year
```

---

### GEELandsatFTV

**Source:** `forestvision/datasets/geelandsat.py`

Fit-to-Vertex (FTV) Harmonized Landsat dataset using LandTrendr temporal segmentation. Provides gap-filled, smoothed time series data.

#### Dataset Information

| Property | Value |
|----------|-------|
| **Algorithm** | LandTrendr FTV |
| **Resolution** | 30 meters |
| **Time Window** | 20-year lookback |
| **NoData** | -32768 |

#### Available Bands

```python
all_bands = ["B1", "B2", "B3", "B4", "B5", "B7", "TCW", "TCG", "TCB", "TCA"]
rgb_bands = ["B5", "B4", "B3"]
```

#### Constructor

```python
GEELandsatFTV(
    year: int,
    roi: BoundingBox,
    season: str = "leafon",
    bands: Optional[List[str]] = None,
    spectral_index: str = "NBR",
    spectral_index_only: bool = False,
    path: Optional[str] = None,
    crs: Optional[CRS] = CRS.from_epsg(5070),
    res: float = 30,
    nodata: Optional[int] = None,
    transforms: Optional[Callable] = None,
    download: bool = False,
    overwrite: bool = False,
    cache: bool = True,
)
```

| Parameter | Description |
|-----------|-------------|
| `year` | Target year for FTV data |
| `spectral_index` | `"NBR"`, `"NDVI"`, or `"TC"` for segmentation |
| `spectral_index_only` | Return only index bands |

---

### GEELandTrendr

**Source:** `forestvision/datasets/geelandsat.py`

Performs LandTrendr (Landsat-based detection of Trends in Disturbance and Recovery) temporal segmentation on harmonized Landsat time series.

#### Algorithm Parameters

```python
lt_params = {
    "maxSegments": 6,
    "spikeThreshold": 0.9,
    "vertexCountOvershoot": 3,
    "preventOneYearRecovery": True,
    "recoveryThreshold": 0.25,
    "pvalThreshold": 0.05,
    "bestModelProportion": 0.75,
    "minObservationsNeeded": 6,
}
```

#### Constructor

```python
GEELandTrendr(
    roi: BoundingBox,
    date_start: int | str,
    date_end: int | str,
    season: str = "leafon",
    spectral_index: str = "NBR",
    ftv_bands: Optional[Tuple] = ["B1", "B2", "B3", "B4", "B5", "B7"],
    crs: Optional[CRS] = None,
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `lt_result` | `ee.Image` | LandTrendr segmentation results |
| `ftv_image(year)` | `ee.Image` | FTV image for specific year |
| `append_transform(image, sindex)` | `ee.Image` | Calculate spectral indices |
| `tasseled_cap_transform(image)` | `ee.Image` | Calculate TC components |

#### Usage Example

```python
from forestvision.datasets import GEELandTrendr

# Run LandTrendr analysis
lt = GEELandTrendr(
    roi=roi,
    date_start=2000,
    date_end=2023,
    season="leafon",
    spectral_index="NBR",
)

# Get FTV image for 2020
ftv_2020 = lt.ftv_image(2020)
```

---

### GEELandTrendrDisturbance

**Source:** `forestvision/datasets/geelandsat.py`

LandTrendr disturbance analysis dataset. Generates images showing years since disturbance, magnitude, duration, and rate of change.

#### Dataset Information

| Property | Value |
|----------|-------|
| **Algorithm** | LandTrendr disturbance analysis |
| **Resolution** | 30 meters |
| **Output Bands** | ysd, mag, dur, rate |
| **NoData** | -32768 |

#### Output Bands

| Band | Description |
|------|-------------|
| `ysd` | Years since largest spectral change detected |
| `mag` | Magnitude of the change |
| `dur` | Duration of the change |
| `rate` | Rate of change |

#### Constructor

```python
GEELandTrendrDisturbance(
    year: int,
    roi: BoundingBox,
    date_start: int | str | None = None,
    date_end: int | str | None = None,
    bands: Optional[List[str]] = None,
    season: str = "leafon",
    spectral_index: str = "NBR",
    flip_disturbance: bool = False,
    big_fast: bool = False,
    sieve: bool = False,
    path: Optional[str] = None,
    crs: Optional[CRS] = CRS.from_epsg(5070),
    res: float = 30,
    nodata: Optional[int] = None,
    transforms: Optional[Callable] = None,
    download: bool = False,
    overwrite: bool = False,
    cache: bool = True,
)
```

| Parameter | Description |
|-----------|-------------|
| `year` | Current year for calculating years since disturbance |
| `flip_disturbance` | Flip sign so disturbances show increasing reflectance |
| `big_fast` | Filter for magnitude > 100 and duration < 4 years |
| `sieve` | Filter disturbances affecting < 11 connected pixels |

---

### GEEDynamicWorld

**Source:** `forestvision/datasets/geedw.py`

Dynamic World V1 land cover classification from Google Earth Engine. Near real-time 10m resolution global land use/land cover mapping.

#### Dataset Information

| Property | Value |
|----------|-------|
| **GEE Asset** | `GOOGLE/DYNAMICWORLD/V1` |
| **Resolution** | 10 meters |
| **Citation** | Brown et al., 2022, Sci Data 9, 251 |

#### Land Cover Classes

| Value | Class | Color |
|-------|-------|-------|
| 0 | Water | #419BDF |
| 1 | Trees | #397D49 |
| 2 | Grass | #88B053 |
| 3 | Flooded vegetation | #7A87C6 |
| 4 | Crops | #E49635 |
| 5 | Shrub & Scrub | #DFC35A |
| 6 | Built Area | #C4281B |
| 7 | Bare ground | #A59B8F |
| 8 | Snow & Ice | #B39FE1 |
| 9 | label | Discrete class [0-8] with highest probability |

#### Available Bands

```python
all_bands = [
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice", "label"
]
```

#### Constructor

```python
GEEDynamicWorld(
    date_start: str,
    date_end: str,
    roi: Optional[BoundingBox] = None,
    res: float = 10,
    class_name: Optional[str] = None,
    bands: Optional[str] = None,
    path: Optional[str] = None,
    crs: Optional[CRS] = CRS.from_epsg(5070),
    transforms: Optional[Callable] = None,
    download: bool = False,
    overwrite: bool = False,
    cache: bool = True,
)
```

#### Usage Example

```python
from forestvision.datasets import GEEDynamicWorld

# Get land cover for 2023
dw = GEEDynamicWorld(
    date_start="2023-01-01",
    date_end="2023-12-31",
    roi=roi,
    bands=["label"],  # Discrete land cover class
)

sample = dw[roi]
# sample["mask"] contains land cover classes 0-8
```

---

### GEEAlphaEarth

**Source:** `forestvision/datasets/geealphaearth.py`

Google AlphaEarth Satellite Embeddings - 64-dimensional embedding vectors for each 10m pixel generated by the AlphaEarth Foundations model.

#### Dataset Information

| Property | Value |
|----------|-------|
| **GEE Asset** | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| **Resolution** | 10 meters |
| **Temporal Resolution** | Annual |
| **Dimensions** | 64-dimensional vectors |

#### Available Bands

```python
all_bands = [f"A{i:02d}" for i in range(64)]  # A00-A63
rgb_bands = ["A01", "A16", "A09"]  # For visualization
```

#### Constructor

```python
GEEAlphaEarth(
    year: int,
    roi: Optional[BoundingBox] = None,
    res: float = 10,
    bands: Optional[Tuple[str, ...]] = None,
    path: Optional[str] = None,
    crs: Optional[CRS] = CRS.from_epsg(5070),
    transforms: Optional[Callable] = None,
    download: bool = False,
    overwrite: bool = False,
    cache: bool = True,
)
```

#### Key Features

- Unit-length vectors distributed on a sphere
- Consistency across years enables change detection
- Robust to clouds and missing data
- Ready for classification, regression, and change detection

---

### GEE3Dep

**Source:** `forestvision/datasets/gee3dep.py`

USGS 3DEP 10m National Map Seamless Digital Elevation Model (DEM).

#### Dataset Information

| Property | Value |
|----------|-------|
| **GEE Asset** | `USGS/3DEP/10m_collection` |
| **Resolution** | 10.2 meters |
| **Coverage** | Contiguous U.S., Hawaii, U.S. territories |
| **Citation** | U.S. Geological Survey, 3D Elevation Program |

#### Available Bands

```python
all_bands = ["elevation"]  # Elevation in meters
```

#### Constructor

```python
GEE3Dep(
    roi: Optional[BoundingBox] = None,
    res: float = 10.2,
    bands: Optional[list] = None,
    path: Optional[str] = None,
    crs: Optional[CRS] = CRS.from_epsg(5070),
    transforms: Optional[Callable] = None,
    download: bool = False,
    overwrite: bool = False,
    cache: bool = True,
)
```

---

### GEEGlobalForestChange

**Source:** `forestvision/datasets/geegfc.py`

University of Maryland's Global Forest Change dataset from Google Earth Engine (2000-2023).

#### Dataset Information

| Property | Value |
|----------|-------|
| **GEE Asset** | `UMD/hansen/global_forest_change_2023_v1_11` |
| **Resolution** | ~30 meters (1 arc-second) |
| **Citation** | Hansen et al., 2013, Science |
| **NoData** | 0 |

#### Available Bands

```python
all_bands = [
    "treecover2000",  # Tree cover circa 2000, range [0, 100]
    "loss",           # Loss 2000-2023, values 0 or 1
    "gain",           # Gain 2000-2012, values 0 or 1
    "lossyear",       # Year of loss, range [0,23]
    "first_b30",      # Landsat band 3 t1
    "first_b40",      # Landsat band 4 t1
    "first_b50",      # Landsat band 5 t1
    "first_b70",      # Landsat band 6 t1
    "last_b30",       # Landsat band 3 t2
    "last_b40",       # Landsat band 4 t2
    "last_b50",       # Landsat band 5 t2
    "last_b70",       # Landsat band 6 t2
    "datamask",       # Data mask (nodata/land/water)
]
```

#### Constructor

```python
GEEGlobalForestChange(
    roi: Optional[BoundingBox] = None,
    res: float = 30,
    bands: Optional[list] = None,
    path: Optional[str] = None,
    crs: Optional[CRS] = CRS.from_epsg(5070),
    transforms: Optional[Callable] = None,
    download: bool = False,
    overwrite: bool = False,
    cache: bool = True,
)
```

---

## Local Raster Datasets

### eMapRAGB

**Source:** `forestvision/datasets/emapragb.py`

eMapR Aboveground Biomass estimates for the Contiguous United States (CONUS) from 1990 to 2018.

#### Dataset Information

| Property | Value |
|----------|-------|
| **Units** | Mg/ha (Megagrams per hectare) |
| **Resolution** | 30 meters |
| **CRS** | EPSG:5070 |
| **NoData** | -32768 |
| **Citation** | Hooper & Kennedy, 2018, Remote Sensing of Environment |

#### Constructor

```python
eMapRAGB(
    paths: Path | Iterable[Path] = "data/datasets/emapr",
    year: Optional[int] = None,
    crs: Optional[CRS] = None,
    res: Optional[float] = 30,
    transforms: Optional[Callable] = None,
    cache: bool = False,
)
```

| Parameter | Description |
|-----------|-------------|
| `paths` | Directory containing `*_cog.tif` files |
| `year` | Optional year filter (filters by filename) |

#### Usage Example

```python
from forestvision.datasets import eMapRAGB

# Load biomass data for 2020
biomass = eMapRAGB(
    paths="data/emapr",
    year=2020,
)

sample = biomass[roi]
# sample["mask"] contains biomass values in Mg/ha
```

---

### ForestOwnership

**Source:** `forestvision/datasets/forestown.py`

USFS Forest Ownership circa 2017 depicting eight ownership categories across the conterminous United States.

#### Dataset Information

| Property | Value |
|----------|-------|
| **Resolution** | 30 meters |
| **CRS** | EPSG:6269 |
| **NoData** | 0 |
| **Citation** | Sass et al., 2020, USDA Forest Service |

#### Ownership Categories

| Value | Category |
|-------|----------|
| 1 | Family |
| 2 | Corporate |
| 3 | TIMO/REIT |
| 4 | Other Private |
| 5 | Federal |
| 6 | State |
| 7 | Local |
| 8 | Tribal |

#### Constructor

```python
ForestOwnership(
    paths: Path | Iterable[Path] = "data/datasets/forest_own1",
    crs: Optional[CRS] = None,
    res: Optional[float] = 30,
    transforms: Optional[Callable] = None,
    cache: bool = False,
)
```

---

### GNNForestAttr

**Source:** `forestvision/datasets/osugnn.py`

Oregon State University's Gradient Nearest Neighbor (GNN) forest attributes data (2021).

#### Dataset Information

| Property | Value |
|----------|-------|
| **Resolution** | 30 meters |
| **CRS** | EPSG:5070 |
| **NoData** | -2147483648 (updated to -999 if remapping) |
| **Attribution** | LEMMA Team, 2020 |

#### Available Bands

| Band | Description |
|------|-------------|
| `fortypba` | Forest type (requires remapping) |
| `cancov` | Canopy cover (0-10,000) |
| `stndhgt` | Height of dominant/co-dominant trees |
| `mndbhba` | Basal-area-weighted average dbh |
| `qmd_dom` | Quadratic mean diameter of dominant trees |
| `ba_ge_3` | Basal area of trees >2.5cm dbh (m2/ha) |
| `tph_ge_3` | Trees per hectare >2.5cm dbh |
| `bph_ge_3_crm` | Biomass of trees >2.5cm dbh (kg/ha) |
| `cancov_layers` | Number of canopy cover layers |

#### Constructor

```python
GNNForestAttr(
    paths: Path | Iterable[Path] = "data/datasets/gnn",
    bands: Sequence[str] = ["fortypba"],
    remap: bool = True,
    crs: Optional[CRS] = None,
    res: Optional[float] = 30,
    transforms: Optional[Callable] = None,
    cache: bool = False,
)
```

| Parameter | Description |
|-----------|-------------|
| `bands` | List of bands to load |
| `remap` | Remap forest type codes using built-in dictionary |

#### Forest Type Remapping

The `remap_dict` attribute maps GNN forest type codes to Oregon Department of Forestry (ODF) standard codes.

---

## Utility Classes

### GPDFeatureCollection

**Source:** `forestvision/datasets/vector.py`

Helper class for handling GeoDataFrame objects as TorchGeo-compatible feature collections.

#### Constructor

```python
GPDFeatureCollection(
    tiles: str | GeoDataFrame,
    sample: int = None,
    crs: CRS = None,
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `bounds` | `BoundingBox` | Total bounds of all features |
| `shape` | `tuple` | Shape of the GeoDataFrame |
| `crs` | `CRS` | Coordinate reference system |

#### Methods

| Method | Description |
|--------|-------------|
| `split(**kwargs)` | Train/test split using `sklearn.model_selection.train_test_split` |
| `plot(**kwargs)` | Plot the GeoDataFrame |

#### Usage Example

```python
from forestvision.datasets import GPDFeatureCollection

# Load tiles from GeoJSON
tiles = GPDFeatureCollection(
    tiles="data/tiles/training_tiles.geojson",
    crs="EPSG:5070",
)

# Get a bounding box for first tile
bbox = tiles[0]

# Split into train/val
train, val = tiles.split(test_size=0.2, random_state=42)
```

---

### DatasetStats

**Source:** `forestvision/datasets/utils.py`

Computes dataset statistics (mean, std, min, max) for normalization.

#### Constructor

```python
DatasetStats(
    dataset: GeoDataset,
    sampler: GeoSampler,
    path: str | Path = None,
    collate_fn: Callable = stack_samples,
    channels: int = None,
    nodata: int = None,
    on_dims: Tuple[int, ...] = (0, 2, 3),
    batch_size: int = 1,
    num_workers: int = 1,
    overwrite: bool = False,
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `compute()` | `dict[str, torch.Tensor]` | Compute and return statistics |

#### Returns Dictionary

```python
{
    "mean": torch.Tensor,           # Per-channel mean
    "std": torch.Tensor,            # Per-channel standard deviation
    "min": torch.Tensor,            # Per-channel minimum
    "max": torch.Tensor,            # Per-channel maximum
    "nodata": int,                  # NoData value
    "nodata_pixels": str,           # Count and percentage
    "sample_size": int,             # Number of samples processed
}
```

#### Usage Example

```python
from forestvision.datasets.utils import DatasetStats
from torchgeo.samplers import GridGeoSampler

# Create sampler
sampler = GridGeoSampler(dataset, size=256, stride=256)

# Compute statistics
stats_calculator = DatasetStats(
    dataset=dataset,
    sampler=sampler,
    path="data/stats.pt",
    batch_size=4,
    num_workers=4,
)
stats = stats_calculator.compute()

# Use for normalization
mean = stats["mean"]
std = stats["std"]
```

---

## Helper Functions

### minmax_scaling

```python
minmax_scaling(data: torch.Tensor, nodata: float) -> torch.Tensor
```

Apply min-max scaling to a multi-dimensional tensor with shape CxHxW.

### save_cog

```python
save_cog(
    data: numpy.ndarray,
    profile: dict,
    path: str,
    overwrite: bool = False,
    window: Window = None,
) -> None
```

Save data as a Cloud Optimized GeoTIFF (COG).

### hash_bbox

```python
hash_bbox(bbox: BoundingBox) -> str
```

Generate a short MD5 hash from a bounding box for unique identification.

### valid_date

```python
valid_date(date: str) -> Tuple[str, str]
```

Validate and parse a date string in YYYY-MM-DD format.

---

## Common Workflows

### Combining Datasets for Training

```python
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from forestvision.datasets import GEESentinel2, GNNForestAttr
from forestvision.samplers import BalancedGridGeoSampler

# Define ROI
roi = BoundingBox(-122.5, -122.4, 45.5, 45.6, 0, 1e12)

# Create imagery dataset
sentinel = GEESentinel2(
    year=2023,
    roi=roi,
    bands=["B4", "B3", "B2", "B8"],
    path="./data/sentinel",
    download=True,
)

# Create target dataset
gnn = GNNForestAttr(
    paths="data/gnn",
    bands=["fortypba", "cancov"],
    remap=True,
)

# Combine datasets (requires custom dataset or datamodule)
# See forestvision.datamodules for pre-built solutions
```

### Computing Normalization Statistics

```python
from forestvision.datasets.utils import DatasetStats
from torchgeo.samplers import GridGeoSampler

sampler = GridGeoSampler(sentinel, size=256, stride=128)
stats = DatasetStats(
    dataset=sentinel,
    sampler=sampler,
    path="sentinel_stats.pt",
).compute()
```

---

## References

1. [TorchGeo Documentation](https://torchgeo.readthedocs.io/)
2. [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets)
3. [eMapR Lab](https://emapr.ceoas.oregonstate.edu/)
4. [LEMMA - Landscape Ecology Modeling, Mapping, and Analysis](https://lemma.forestry.oregonstate.edu/)
5. [Hansen Global Forest Change](https://earthenginepartners.appspot.com/science-2013-global-forest)
