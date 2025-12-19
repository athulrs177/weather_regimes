# Weather Regime Analysis

This repository contains the Python-based scritps for computing and detecting weather regimes as explained in Grams et al. 2017 (Balancing Europe’s wind-power output through spatial deployment informed by weather regimes).

![Weather Regimes DJF](weather_regimeDJF.png)

## Requirements
- Python 3.9+
- NumPy, SciPy, xarray, scikit-learn, pandas 
- matplotlib, cartopy (for plotting)

## Usage
All necessary functions to compute the 6h climatology, compute 10-day low pass filter etc are kept in helper_functions.py. DO NOT change these.
The cluster means neceaasry for the projections of any general dataset is already computed using ERA5 Z500 dataset from 1979-2015, as described in Grams et al. 2017.
Any general Z500 anomaly field, model-based or observation based, for any general time period can now be projected onto these cluster means.

If the cluster means need to be computed from scratch for any general dataset, please uncomment the necessary section.
