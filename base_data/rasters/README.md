# Raster Data Directory

This directory contains the baseline raster datasets (`.tif` format) required for `GMFAgent` spatial analysis.

## 1. Regional Cropping Notice
Due to GitHub's file size limitations (100MB per file), the `.tif` files provided in this repository have been **cropped to the Japan and East Asia region** (approx. 128°E - 146°E, 30°N - 46°N). 

These files are intended for demonstration and regional testing purposes.

## 2. Global Data & Licensing
If you require global coverage or different regions, please download the original datasets from the official providers listed below. 

> **Important:** When downloading and using these datasets, you must adhere to the specific **terms of use and licenses** provided by each organization.

* **Population Data:** [Download from https://landscan.ornl.gov/]  
    *License: Please refer to the provider's licensing terms.*
* **Vs30 (Global Shear-Wave Velocity):** [Download from https://earthquake.usgs.gov/data/vs30/]  
    *License: Please refer to the provider's licensing terms.*
* **DEM (Digital Elevation Model):** [Download from https://www.ngdc.noaa.gov/mgg/topo/globe.html]  
    *License: Please refer to the provider's licensing terms.*

## 3. Extending GMFAgent
`GMFAgent` is designed to be highly extensible. Users are encouraged to integrate additional raster layers to perform more comprehensive multi-hazard or socio-economic analyses.

To add your own data:
1. Place your `.tif` files (e.g., land use, building density, or infrastructure maps) into this `base_data/rasters/` folder.
2. Ensure the CRS (Coordinate Reference System) matches your study area (WGS84 is recommended).
3. Update your local configuration to include these new layers in the analysis pipeline.
