## U-DEM

Workflow:
- Download and preprocess Sentinel-1, ArcticDEM and CryoSat-2 data: `data-preprocessing.ipynb` (refers further to these scripts: `download-arcticdem.py`, `preprocessing_tools.py`, and `reviewingArcticDEM.py`
- Download and preprocess ICESat-2 validation data: `download-icesat2.ipynb`
- Do hyperparameter optimization: `run-optuna.py`
