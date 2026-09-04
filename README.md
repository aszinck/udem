## U-DEM

Workflow:
- Download and preprocess Sentinel-1, ArcticDEM and CryoSat-2 data: `data-preprocessing.ipynb` (refers further to these scripts: `download-arcticdem.py`, `reviewingArcticDEM.py`, and `preprocessing_tools.py`)
- Download and preprocess ICESat-2 validation data: `download-icesat2.ipynb`
- Do hyperparameter optimization: `run-optuna.py` (also uses `udem.py`)
- Do model training: `training-udem.py` (also uses `udem.py`)
- Make predictions: `predict-udem.ipynb`(also uses `udem.py` and `validation_tools.py`)
- Validate U-DEM: `validate-udem.ipynb`(also uses `validation_tools.py`)
- Apply U-DEM to surging glaciers: `surgingglaciers.ipynb`
- Apply U-DEM to subglacial lakes: `subglacial-lakes.ipynb`
