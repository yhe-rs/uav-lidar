# LiDAR Point Cloud Processing & AGB Estimation Framework

This repository provides a high-performance, dual-language (**R and Python**) framework for the end-to-end processing of Airborne Laser Scanning (ALS) data. The workflow bridges raw LiDAR point cloud geometry with advanced machine learning to deliver high-resolution forest **Aboveground Biomass (AGB)** estimations.

---

### 🌲 Core Workflow Modules

#### 1. LiDAR Point Cloud Processing & Feature Extraction
Utilizing a hybrid approach, this module focuses on transforming raw `.las` / `.laz` data into ecologically meaningful metrics:
* **Normalization & DTM Generation:** Automated filtering of ground points and interpolation of Digital Terrain Models (DTM).
* **Canopy Height Model (CHM) Generation:** High-resolution CHM creation using pit-free algorithms to minimize artifacts in dense canopy cover.
* **Individual Tree Segmentation (ITS):** * Implementation of **Watershed Segmentation** and local maximum filters to identify tree tops.
    * Delineation of Individual Tree Crowns (ITC) to extract per-tree structural attributes.

#### 2. Advanced Feature Engineering
The workflow extracts high-dimensional descriptors of forest structure to serve as model predictors:
* **Structural Metrics ($z$):** Vertical distribution statistics, for instance mean height, 95th percentile ($P_{95}$), and skewness of the point return distribution.
* **Radiometric Intensity ($i$):** Intensity-based metrics to differentiate between species types and assess moisture content or canopy health.
* **Voxel-based Analysis:** Optional 3D voxelization to quantify sub-canopy density and vertical complexity.

#### 3. Machine Learning-Based AGB Mapping
Integrating structural geometry with gradient-boosted decision trees to mitigate common saturation issues:
* **XGBoost Optimization:** Implementation of an **XGBoost** regressor tuned via Bayesian optimization to handle non-linear relationships between LiDAR metrics and field-measured biomass.
* **Multiscale Integration:** Capability to map AGB at both the individual tree level and the plot/landscape level (e.g., 10m or 30m resolution).
* **Explainable AI (XAI):** Integration of **SHAP analysis** to quantify the influence of specific height percentiles versus return intensity on the final biomass estimate.

---

### 🛠️ Technical Implementation
* **Languages:** `R` (primarily using `lidR` and `terra`) and `Python` (using `XGBoost`, `PyTorch`, and `GDAL`).
* **Key Algorithms:** Watershed Segmentation and Extreme Gradient Boosting.
* **Scalability:** Optimized for High-Performance Computing (HPC) environments using parallel processing for large-scale LiDAR tiles.

---

### 📝 Summary
The workflow transforms raw multi-return LiDAR data (DJI L1/L2) into structural features, segments individual trees via statistical spatial testing, and employs a tuned XGBoost model to interpret and map the factors driving forest biomass.

<img width="1024" height="1536" alt="uav-lidar-agb" src="https://github.com/user-attachments/assets/b7c4ce3f-e09b-48df-bf16-77c441325e80" />

