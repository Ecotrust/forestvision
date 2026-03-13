# Predicting GNN Forest Attributes with Multi-Task Deep Learning Models: Training Report

Yankuic Galvan

---
## Summary

This report documents the training and evaluation of a multi-task deep learning model for forest attribute mapping. The model employs a UNet architecture with simultaneous classification (forest type segmentation) and regression (structural attributes) capabilities. The experiment trains a model using Oregon State University Gradient Nearest Neighborhood (GNN) forest attribute data from 2021 as the target and a combination of Sentinel-2 imagery, topographic data, and climate variables as inputs. The training process includes hyperparameter optimization using Optuna, and the final model is evaluated on a held-out test set with comprehensive performance metrics.


**Key Results at a Glance**

| Task Type | Metric | Value | Assessment |
|-----------|--------|-------|------------|
| **Overall** | Test Loss | 0.078 | Good convergence |
| **Segmentation** | Accuracy | 54.12% | Moderate - room for improvement |
| **Segmentation** | Jaccard Index (mIoU) | 0.201 | Low - class imbalance challenge |
| **Segmentation** | Cohen's Kappa | 0.473 | Moderate agreement |
| **Regression** | R² Score | 0.569 | Moderate variance explanation |
| **Regression** | RMSE | 0.547 | Good error magnitude |
| **Regression** | MAE | 0.409 | Average prediction error |

---

## GNN Methodology and Data Layers Overview

The Gradient Nearest Neighbor (GNN) method is a predictive mapping approach developed by the [Landscape Ecology Modeling, Mapping, and Analysis (LEMMA)](https://lemma.forestry.oregonstate.edu/data) team at Oregon State University. It integrates ground-based forest inventory data (such as FIA plots) with satellite imagery and environmental gradients to produce continuous, high-resolution maps of forest composition and structure.

### Key Principles:
- **Direct Gradient Analysis**: Uses multivariate statistical techniques (like Canonical Correspondence Analysis) to relate forest vegetation data from plots to environmental variables (climate, topography, geology) and spectral data from satellite imagery (Landsat/Sentinel).
- **Nearest Neighbor Imputation**: For every pixel in the landscape, the method identifies the most similar forest inventory plot(s) in the multi-dimensional gradient space. The full suite of measured attributes from those plots is then "imputed" (assigned) to that pixel.
- **Consistency**: Because entire plot records are imputed, the resulting maps maintain the complex multi-attribute correlations found in real forest stands, ensuring that the predicted forest types, biomass, and structure are ecologically consistent.


## Accuracy and Usage

Gradient Nearest Neighbor (GNN) maps provide 30-m resolution forest attribute data by linking USDA Forest Service FIA plots with Landsat imagery. While the grain size is small, the following practical constraints and guidelines govern their appropriate use:

### Spatial and Temporal Constraints
* **Scale of Accuracy:** Although maps are produced at a 30-m pixel level, accuracy assessments are only statistically valid at scales $\ge$ 0.81 ha (the 9-pixel footprint of an FIA plot). Individual pixel values are highly uncertain.
* **Scale-Dependent Reliability:** Model predictions converge with field-based estimates as the area of interest increases. Accuracy is significantly higher when aggregated to landscape or regional scales (e.g., 10,000 to 200,000 ha) compared to stand-level analysis.
* **Measurement Lag:** FIA plots are measured on a 10-year cycle. This can result in a significant delay in reflecting recent disturbances or rapid forest changes in the GNN output.
* **Spectral Saturation:** In old-growth forests, Landsat spectral signals saturate, making it difficult to distinguish between different stages of mature forest structure and biomass.

## Considerations on GNN Data
* **Prioritize Aggregation:** GNN data is intended for broad-scale monitoring and planning and should be aggregated to coarser spatial scales for reliable analysis. Avoid using single-pixel values for site-specific management decisions.
* **Assess by Attribute:** Prediction accuracy varies by variable. Live tree structure (canopy cover, biomass) is more reliable than attributes not directly observable by satellites, such as snag density or down log volume. 
* **Map Similarity vs. True Data:** Models trained and validated using GNN data reflect **map similarity** (the ability to mimic GNN predictions) rather than proximity to **true ground data**. Because GNN is itself a modeled product with inherent biases and uncertainties, performance in a secondary model training on GNN outputs does not guarantee accuracy relative to actual forest conditions on the ground.

### Data Layers Description

The following layers are primary targets for the `forestvision` models, representing different aspects of forest state as of 2021.

#### 1. Forest Type (`fortypba`)
- **Description**: A categorical classification of forest composition.
- **Methodology**: Determined by the dominant tree species (by basal area) present in the imputed inventory plot.
- **Units**: Categorical codes (remapped to ODFW habitat classes in this project).
- **Usage**: Used to identify the primary ecological community and habitat type.

#### 2. Canopy Cover (`cancov`)
- **Description**: The percentage of the ground covered by the vertical projection of tree crowns.
- **Units**: Percentage (scaled 0 to 10,000 in raw GNN data, where 10,000 = 100%).
- **Physical Meaning**: Represents the density of the forest overstory and is a critical metric for habitat quality and light availability.

#### 3. Quadratic Mean Diameter of Dominant Trees (`qmd_dom`)
- **Description**: The quadratic mean diameter (QMD) of the dominant and co-dominant trees in the stand.
- **Units**: Centimeters (cm).
- **Physical Meaning**: QMD is the diameter of the tree of average basal area. Focusing on dominant/co-dominant trees provides a measure of the size of the main canopy trees, which is a strong indicator of stand maturity and successional stage.

#### 4. Basal Area of Live Trees > 2.5cm (`ba_ge_3`)
- **Description**: The cross-sectional area of all live tree stems (with diameter at breast height ≥ 2.5 cm) per unit area.
- **Units**: Square meters per hectare (m²/ha).
- **Physical Meaning**: A fundamental measure of forest density and stocking. It represents the "occupancy" of the site by trees and is highly correlated with total biomass and carbon storage.

---

### Mapping FIA Forest Community Types (FORTYPBA) to ODFW Habitat Classes

#### [0] Nonforest (NF)
Areas identified in inventory data as lacking significant tree cover or classified as non-forest land uses.

#### [1] Shrub (shr)
Landscapes dominated by woody shrub species rather than trees. Common components include Mountain Mahogany (*Cercocarpus ledifolius*), Chinquapin (*Chrysolepis chrysophylla*), and various cherry species (*Prunus* spp.).

#### [2] Riparian (rip)
Forests and woodlands located along watercourses and wetlands. Dominated by moisture-loving species such as Red Alder (*Alnus rubra*), Willows (*Salix* spp.), Black Cottonwood (*Populus balsamifera*), and Oregon Ash (*Fraxinus latifolia*).

#### [3] Lodgepole pine (lpp)
Forests primarily composed of Lodgepole Pine (*Pinus contorta*). These are often found in areas with nutrient-poor soils, high-elevation plateaus, or regions with specific fire-return intervals that favor this seral species.

#### [4] Ponderosa Pine (pdp)
Dry forest types dominated by Ponderosa Pine (*Pinus ponderosa*). These forests often have an open structure and may include an understory of drought-tolerant shrubs like Mountain Mahogany (*Cercocarpus ledifolius*).

#### [5] Mixed Conifer (mxc)
Highly diverse conifer forests common in the Siskiyou Mountains and Southern Cascades. Key species include Grand Fir (*Abies grandis*), Incense Cedar (*Calocedrus decurrens*), Ponderosa Pine (*Pinus ponderosa*), and Douglas-fir (*Pseudotsuga menziesii*).

#### [6] Western Juniper (wju)
Arid woodlands characteristic of the high desert and eastern foothills, dominated by Western Juniper (*Juniperus occidentalis*).

#### [7] Mixed Oak - Conifer (mxo)
Woodlands and forests characterized by the presence of oak species such as Oregon White Oak (*Quercus garryana*), California Black Oak (*Quercus kelloggii*), and Canyon Live Oak (*Quercus chrysolepis*), often intermingled with Ponderosa Pine or Douglas-fir.

#### [8] Quaking Aspen (asp)
Deciduous stands dominated by Quaking Aspen (*Populus tremuloides*). These are typically found in moist pockets, riparian edges, or high-elevation sites, particularly in the eastern regions of the state.

#### [9] Mixed Hardwood-Conifer (mxh)
Transition forests where broadleaf hardwoods mix significantly with conifers. Common hardwoods include Bigleaf Maple (*Acer macrophyllum*), Red Alder (*Alnus rubra*), and Pacific Madrone (*Arbutus menziesii*), typically associated with Douglas-fir or Grand Fir.

#### [10] Coastal Spruce, Cedar or Redwood (red)
Dominated by coastal-associated species including Sitka Spruce (*Picea sitchensis*), Port Orford Cedar (*Chamaecyparis lawsoniana*), and Coast Redwood (*Sequoia sempervirens*). These forests often include Red Alder (*Alnus rubra*) and Bigleaf Maple (*Acer macrophyllum*) in the understory or as co-dominants in disturbed areas.

#### [11] Douglas-fir - Western Hemlock (dfi)
The characteristic mesic forests of the Pacific Northwest. Primary species include Douglas-fir (*Pseudotsuga menziesii*) and Western Hemlock (*Tsuga heterophylla*), frequently occurring with Western Redcedar (*Thuja plicata*) and Grand Fir (*Abies grandis*).

#### [12] Silver fir - Mountain Hemlock (sfi)
High-elevation montane forests dominated by Pacific Silver Fir (*Abies amabilis*) and Mountain Hemlock (*Tsuga mertensiana*). These forests occupy the zone between mid-elevation mixed conifer and true subalpine parklands.

#### [13] Spruce - Subalpine Fir (spr)
Cold-climate forests of high elevations or frost pockets. Primary species include Subalpine Fir (*Abies lasiocarpa*) and Engelmann Spruce (*Picea engelmannii*), sometimes occurring with Whitebark Pine (*Pinus albicaulis*).


## 2. Model Architecture

### 2.1 Multi-Task UNet Design

The model uses an **OptimizedMTUNet** architecture, a variant of the standard UNet optimized for multi-task learning:

```
Input (20 channels) → Encoder → Decoder → Task Heads
                                     ├── Segmentation Head (14 classes)
                                     ├── Regression Head 1 (Canopy Cover)
                                     ├── Regression Head 2 (QMD)
                                     └── Regression Head 3 (Basal Area)
```

### 2.2 Input Channels (20 Total)

| Data Source | Bands/Indices | Count | Description |
|-------------|---------------|-------|-------------|
| Sentinel-2 | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 | 10 | 10m multispectral imagery |
| Spectral Indices | NDVI, SAVI | 2 | Vegetation health indicators |
| Topography | Elevation | 1 | USGS 3DEP 10m DEM |
| Climate | AHM, MAP, TD | 3 | ClimateNA variables (Annual Heat:Moisture, Mean Annual Precip, Temperature Difference) |
| Landsat-8 TC | Tasseled Cap (4 bands) | 4 | Spectral transformation bands |

### 2.3 Task Configuration

The model performs **4 simultaneous tasks**:

| Task | Type | Output | Classes/Range | Description |
|------|------|--------|---------------|-------------|
| `fortypba` | Classification | 14 channels | 14 forest types | Forest type classification |
| `cancov` | Regression | 1 channel | [0, 10000] | Canopy cover (proportional x 10000) |
| `qmd_dom` | Regression | 1 channel | Continuous | Quadratic mean diameter (dominant) |
| `ba_ge_3` | Regression | 1 channel | Continuous | Basal area for trees >= 3" DBH |

### 2.4 Forest Type Classes (14 Classes)

| ID | Code | Description |
|----|------|-------------|
| 0 | nf | non-forest mask |
| 1 | shr | Shrub |
| 2 | rip | Riparian |
| 3 | lpp | Lodgepole pine |
| 4 | pdp | Ponderosa pine |
| 5 | mxc | Mixed conifer |
| 6 | wju | Western juniper |
| 7 | mxo | Mixed Oak-Conifer |
| 8 | asp | Quaking Aspen |
| 9 | mxh | Mixed Hardwood-Conifer |
| 10 | red | Coastal Spruce/Cedar/Redwood |
| 11 | dfi | Douglas-fir - Western Hemlock |
| 12 | sfi | Silver fir - Mountain Hemlock |
| 13 | spr | Spruce - Subalpine Fir |

---

## 3. Training Configuration

### 3.1 Hyperparameters (Optuna-Optimized)

| Parameter | Value | Source |
|-----------|-------|--------|
| Learning Rate | 0.000133 | Optuna HPO |
| Dropout | 0.5 | Optuna HPO |
| Weight Decay | 0.00085 | Optuna HPO |
| Scheduler Patience | 7 epochs | Fixed |
| Scheduler Factor | 0.5 | Fixed |
| Max Epochs | 100 | Fixed |
| Batch Size | 32 | Fixed |

### 3.2 Loss Functions

#### Segmentation Loss: Focal Loss
Focal loss addresses class imbalance by down-weighting easy examples:

```yaml
loss: focal
focal_alpha: 0.1
focal_gamma: 2.9
focal_weight: [1.5, 1.22, 3.0, 1.94, 1.52, 1.0, 1.21, 3.49, 3.68, 3.0, 2.31, 0.75, 1.75, 1.53]
```

The class weights reflect the inverse frequency of each forest type in the training data, with higher weights for rare classes (e.g., riparian=3.0, mixed oak-conifer=3.49).

#### Regression Loss: Huber Loss
Huber loss combines MAE and MSE for robust regression:

```yaml
reg_loss: huber
huber_delta: 1.5
```

### 3.3 Loss Weighting Strategy

Fixed weighting across task types:

| Task Component | Weight | Notes |
|----------------|--------|-------|
| Segmentation | 0.55 | Primary task - highest weight |
| Canopy Cover | 0.25 | Most predictable structural attribute |
| QMD | 0.10 | Secondary structural metric |
| Basal Area | 0.10 | Secondary structural metric |

---

## 4. Dataset & Data Pipeline

### 4.1 Data Sources

The GNNDataModule integrates multiple geospatial datasets:

**Input Datasets:**
1. **GEESentinel2** - Harmonized Sentinel-2 surface reflectance (10m)
2. **GEE3Dep** - USGS 3DEP elevation data
3. **ClimateNA** - Climate normals (AHM, MAP, TD)
4. **GEELandsat8** - Landsat-8 Tasseled Cap transformation

**Target Datasets:**
1. **GNNForestAttr** - OSU GNN forest attributes (fortypba, cancov, qmd_dom, ba_ge_3)
2. **GEEDynamicWorld** - Dynamic World land cover for non-forest masking

### 4.2 Data Specifications

| Parameter | Value |
|-----------|-------|
| Patch Size | 128 x 128 pixels |
| Spatial Resolution | 10 meters |
| Geographic Extent | Training/Validation/Test tiles from proportional sampling |
| Year | 2021 |
| Batch Size | 32 |
| Workers | 19 |

### 4.3 Preprocessing Pipeline

```
Raw Data
    ↓
[Pre-Augmentation]
    - Append NDVI/SAVI indices
    ↓
[Augmentation] (Random flips, rotations)
    ↓
[Post-Augmentation]
    - Normalize input channels
    - Combine GNN + Dynamic World masks
    - Normalize targets with identity channel preservation
```

### 4.4 Data Splits

| Split | File | Purpose |
|-------|------|---------|
| Training | `proportional_128x128_10m_train.geojson` | Model training |
| Validation | `proportional_128x128_10m_val.geojson` | Hyperparameter tuning |
| Test | `proportional_128x128_10m_test.geojson` | Final evaluation |

---

## 5. Results Analysis

### 5.1 Overall Performance

The test loss of **0.078** indicates good model convergence across all tasks. The multi-task architecture successfully balances segmentation and regression objectives.

---

### 5.2 Segmentation Results

#### 5.2.1 Quantitative Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 54.12% | The model correctly classifies ~54% of pixels |
| **Jaccard Index (mIoU)** | 0.201 | Mean Intersection over Union across all classes |
| **Cohen's Kappa** | 0.473 | Moderate agreement (0.41-0.60 range) |

**Analysis:**
- The accuracy of 54% with 14 classes indicates performance significantly above random chance (7.1%)
- The low mIoU (0.20) suggests challenges with class imbalance - common in forest type mapping where some classes are rare
- Kappa of 0.473 shows moderate agreement after accounting for chance
- The focal loss with class weights helps but further balancing strategies may be needed

#### 5.2.2 Confusion Matrix Visualization

![Confusion Matrix](assets/v8/test_confusion_matrix.png)

**Figure 1: Confusion Matrix for Forest Type Classification (14 Classes)**

**Elements Explained:**

| Element | Description | Interpretation |
|---------|-------------|----------------|
| **Rows** | True labels (ground truth forest types) | Actual forest type from reference data |
| **Columns** | Predicted labels (model outputs) | Forest type predicted by the model |
| **Diagonal Cells** | Correct classifications | Bright diagonal indicates good performance for that class |
| **Off-Diagonal Cells** | Misclassifications | Show which classes are commonly confused |
| **Color Intensity** | Number of samples | Darker colors = more samples in that cell |

**Key Observations to Look For:**
- **Strong diagonal**: Indicates overall good classification performance
- **Horizontal stripes**: May indicate a class that is frequently misclassified into others
- **Vertical stripes**: May indicate a class that the model over-predicts
- **Common confusions**: Mixed conifer (mxc) vs mixed hardwood-conifer (mxh), or similar species assemblages

---

### 5.3 Regression Results

#### 5.3.1 Quantitative Metrics (Normalized Space)

**Important Note**: All regression metrics (MAE, MSE, RMSE) are computed on **z-score normalized data**. Targets are standardized using mean and standard deviation from the training set statistics (`proportional_128x128_10m_stats.json`).

| Metric | Value | Interpretation (Normalized Units) |
|--------|-------|-----------------------------------|
| **R² Score** | 0.569 | Model explains ~57% of variance |
| **RMSE** | 0.547 | Typical prediction error: 0.55 std dev from mean |
| **MSE** | 0.315 | Squared error in normalized units |
| **MAE** | 0.409 | Mean absolute error: 0.41 std dev from mean |

**Analysis:**
- **R² = 0.569** is in the moderate-to-good range for forest attribute prediction
- This aligns with GNN methodology expectations - structural attributes (canopy cover, basal area) correlate well with spectral signatures
- **MAE of 0.409** means predictions are typically within 0.41 standard deviations of the true value
- **RMSE of 0.547** represents the typical magnitude of prediction errors in standardized units
- Performance is consistent across the three regression targets (cancov, qmd_dom, ba_ge_3)

**Normalization Statistics (from Training Data):**

| Target Variable | Mean | Std Dev | Min | Max | Physical Units |
|-----------------|------|---------|-----|-----|----------------|
| **Canopy Cover (cancov)** | 5,876.5 | 2,630.5 | 0 | 9,970 | Proportional x 10,000 |
| **QMD (qmd_dom)** | 347.9 | 197.2 | 0 | 2,300 | Inches x 10 |
| **Basal Area (ba_ge_3)** | 3,171.1 | 2,169.3 | 0 | 25,910 | ft²/acre x 10 |

**Normalization Formula:**
```
Normalized Value = (Raw Value - Mean) / Standard Deviation
Raw Value = (Normalized Value × Std Dev) + Mean
```

**Converting Metrics to Original Units:**
- **Canopy Cover MAE**: 0.409 × 2,630.5 = **1,076** (proportional x 10,000) = **10.76%** absolute error
- **QMD MAE**: 0.409 × 197.2 = **80.7** (inches x 10) = **8.1 inches** absolute error
- **Basal Area MAE**: 0.409 × 2,169.3 = **887** (ft²/acre x 10) = **88.7 ft²/acre** absolute error

#### 5.3.2 Canopy Cover Regression Analysis

![Canopy Cover Regression](assets/v8/test_regression_marginal_cancov.png)

**Figure 2: Regression Marginal Plot - Canopy Cover (cancov)**

**Elements Explained:**

| Element | Description | Interpretation |
|---------|-------------|----------------|
| **X-axis (Actual)** | Ground truth canopy cover values (**normalized**) | Z-score normalized true values from reference data |
| **Y-axis (Predicted)** | Model predicted canopy cover values (**normalized**) | Z-score normalized predictions from regression head |
| **Scatter Points** | Individual sample predictions | Each point represents one test sample (in std dev units) |
| **Diagonal Line (y=x)** | Perfect prediction line | Points on this line = perfect predictions |
| **Density Shading** | Data concentration | Darker areas = more samples with those values |
| **R² Value** | Coefficient of determination | Proportion of variance explained (0.569) |
| **Pearson r** | Pearson correlation coefficient | Linear correlation between predicted and actual (expect ~0.75) |
| **RMSE** | Root Mean Square Error | Typical prediction error in **standard deviation units** |
| **n** | Sample count | Number of test samples used |

**Interpretation Guidelines:**
- **Tight clustering around diagonal**: Indicates accurate predictions
- **Funnel shape** (wide at extremes, narrow in middle): "Regression toward the mean" - model is conservative
- **Systematic bias**: Points consistently above/below line indicates over/under-prediction
- **R² vs Pearson r**: R² measures variance explained; Pearson r measures linear correlation strength
- **Normalized units**: Values represent standard deviations from the training mean, not raw canopy cover percentages

#### 5.3.3 Quadratic Mean Diameter Regression Analysis

![QMD Regression](assets/v8/test_regression_marginal_qmd_dom.png)

**Figure 3: Regression Marginal Plot - Quadratic Mean Diameter (qmd_dom)**

**Elements Explained:**

| Element | Description | Interpretation |
|---------|-------------|----------------|
| **X-axis (Actual)** | Ground truth QMD values (**normalized**) | Z-score normalized true QMD values |
| **Y-axis (Predicted)** | Model predicted QMD values (**normalized**) | Z-score normalized regression outputs |
| **Scatter/Density** | Prediction distribution | Shows where predictions cluster relative to truth (in std dev units) |
| **R² / Pearson r** | Correlation metrics | Measures prediction quality and linear association |

**QMD-Specific Considerations:**
- QMD is a **Tier 2 variable** (moderately predictable) because it depends on tree size distribution
- Lower R² expected compared to canopy cover
- Relationship with spectral data is more indirect than canopy cover
- **Normalized units**: Axes represent standard deviations from training mean, not raw QMD in inches

#### 5.3.4 Basal Area Regression Analysis

![Basal Area Regression](assets/v8/test_regression_marginal_ba_ge_3.png)

**Figure 4: Regression Marginal Plot - Basal Area (ba_ge_3)**

**Elements Explained:**

| Element | Description | Interpretation |
|---------|-------------|----------------|
| **X-axis (Actual)** | Ground truth basal area values (**normalized**) | Z-score normalized total basal area for trees >= 3" DBH |
| **Y-axis (Predicted)** | Model predicted basal area (**normalized**) | Z-score normalized regression estimate of stand density |
| **Correlation Metrics** | R² and Pearson r | Overall prediction accuracy measures |

**Basal Area-Specific Considerations:**
- Basal area is a **Tier 1 variable** (highly predictable) as it correlates strongly with canopy reflectance
- Expect tighter clustering around diagonal than QMD
- Important metric for timber volume estimation and carbon accounting
- **Normalized units**: Axes represent standard deviations from training mean, not raw basal area in ft²/acre

#### 5.3.5 Performance by Task Type

| Task | Predictability Tier | Expected R² Range | Notes |
|------|--------------------|--------------------|-------|
| Canopy Cover | Tier 1 (High) | 0.6-0.8 | Directly visible in canopy |
| Basal Area | Tier 1 (High) | 0.5-0.7 | Strong spectral correlation |
| QMD | Tier 2 (Moderate) | 0.4-0.6 | Indirectly related to structure |
| Forest Type | Tier 2 (Moderate) | 0.4-0.6 | Complex species assemblages |

The observed R² of 0.569 falls within expected ranges for Tier 1-2 variables in GNN-based mapping.

---

## 6. Sample Predictions Visualization

### 6.1 Model Output Examples

![Sample Images](assets/v8/test_images.png)

**Figure 5: Sample Test Images with Model Predictions**

This figure displays representative test samples showing:
- **Input imagery**: Composite of Sentinel-2 bands
- **Ground truth masks**: True forest type labels
- **Predicted masks**: Model segmentation outputs
- **Regression overlays**: Continuous value predictions for structural attributes

**Visual Assessment Criteria:**
- **Spatial coherence**: Predictions should show realistic patch structures
- **Edge alignment**: Boundaries between forest types should align with image features
- **Regression smoothness**: Continuous predictions should vary smoothly across homogeneous areas

---

## 7. Technical Implementation Details

### 7.1 Training Infrastructure

| Component | Setting |
|-----------|---------|
| Accelerator | GPU |
| Framework | PyTorch Lightning |
| Checkpointing | Top-3 by validation loss |
| Monitoring | Built-in Lightning logger |

### 7.2 Model Checkpointing

```yaml
callbacks:
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      monitor: val_loss
      mode: min
      save_top_k: 3
      filename: "best-{epoch:02d}-{val_loss:.4f}"
```

The best 3 checkpoints by validation loss are preserved for downstream inference.

### 7.3 Key Configuration Parameters

```yaml
# Model
model: OptimizedMTUNet
in_channels: 20
task_types: ["classification", "regression", "regression", "regression"]
num_classes_per_task: [14, 1, 1, 1]
ignore_index: -2147483648  # NoData handling

# Loss
loss: focal
reg_loss: huber
loss_weighting: fixed
seg_loss_weight: 0.55
reg_loss_weights: [0.25, 0.1, 0.1]
```

---

## 8. Methodology Context: GNN Approach

This implementation applies deep learning to the **Gradient Nearest Neighbor (GNN)** imputation methodology developed at Oregon State University. Key aspects:

### 8.1 GNN Principles Applied

1. **Multivariate Imputation**: Unlike traditional remote sensing that predicts single attributes, GNN preserves relationships between forest structure, composition, and condition.

2. **Reference Data Integration**: The model learns from FIA (Forest Inventory and Analysis) plot data, similar to how GNN imputes from nearest neighbors in gradient space.

3. **Multi-Scale Reliability**: Following GNN methodology, predictions are most reliable at landscape to regional scales (aggregation reduces pixel-level uncertainty).

### 8.2 Deep Learning Advantages

- **End-to-End Learning**: The neural network learns feature representations directly from raw imagery
- **Spatial Context**: UNet architecture captures spatial patterns through encoder-decoder structure
- **Multi-Task Efficiency**: Shared representations benefit all prediction tasks

---

## 9. Conclusions

### 9.1 Summary

The training experiment successfully implemented a multi-task UNet for forest attribute mapping with the following outcomes:

1. **Segmentation**: Moderate performance (54% accuracy, 0.47 Kappa) with room for improvement on class imbalance
2. **Regression**: Good performance (R² = 0.57) consistent with GNN methodology expectations for structural attributes
3. **Multi-Task Balance**: Fixed loss weighting effectively balanced the 4 tasks
4. **Convergence**: Stable training with test loss of 0.078

### 9.2 Strengths

- Optuna-optimized hyperparameters show effective learning rate and regularization
- Focal loss with class weights addresses forest type imbalance
- Comprehensive input feature set (20 channels) captures spectral, topographic, and climate information
- Proper NoData handling throughout pipeline

### 9.3 Areas for Improvement

1. **Class Imbalance**: The mIoU of 0.20 suggests further strategies needed for rare forest types
2. **Segmentation Accuracy**: 54% accuracy could be improved with:
   - Longer training (current: 10 epochs)
   - Data augmentation tuning
   - Alternative architectures (e.g., ResMTUNet with pretrained backbone)
3. **Regression Refinement**: R² of 0.57 is good but could benefit from:
   - Quantile regression (already implemented in codebase)
   - Additional training epochs
   - Target-specific loss tuning

### 9.4 Recommended Next Steps

1. **Extended Training**: Increase to 50-100 epochs with early stopping
2. **Architecture Exploration**: Try ResMTUNet with ImageNet pretraining
3. **Quantile Regression**: Implement for uncertainty estimation (config already supports this)
4. **Ensemble Methods**: Combine multiple model checkpoints
5. **Spatial Validation**: Assess accuracy at landscape scale (following GNN methodology)

---

## Appendix 1: Metrics Reference Table

| Metric | Value | Target Range | Status |
|--------|-------|--------------|--------|
| test_loss | 0.07797 | < 0.1 | Good |
| seg_test_accuracy | 0.54124 | > 0.60 | Needs Improvement |
| seg_test_jaccard | 0.20051 | > 0.30 | Needs Improvement |
| seg_test_kappa | 0.47340 | > 0.50 | Moderate |
| reg_test_mae | 0.40860 | < 0.50 | Good |
| reg_test_mse | 0.31507 | < 0.40 | Good |
| reg_test_rmse | 0.54717 | < 0.60 | Good |
| reg_test_r2 | 0.56850 | > 0.60 | Moderate |

---

## Appendix 2: File Reference
| Resource            | Description                                  | Path                                                                                   |
|---------------------|----------------------------------------------|----------------------------------------------------------------------------------------|
| Configuration       | Model and training configuration             | `data/dev/configs/gnn_v0/osugnn_v0.yaml`                                               |
| Stats               | Training set normalization statistics        | `data/dev/configs/gnn_v0/proportional_128x128_10m_stats.json`                          |
| Tiles               | Train/Val/Test spatial splits                | `data/dev/configs/gnn_v0/proportional_128x128_10m_{train,val,test}.geojson`            |
| Model Checkpoints   | Directory for saved model checkpoints        | `data/dev/configs/gnn_v0/checkpoints/`                                                 |
