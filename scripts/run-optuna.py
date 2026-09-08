import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.amp import autocast, GradScaler
import os
import pickle
import pandas as pd
from udem import ImageDataset, UNet
import argparse
import gc



# GLOBALS THAT GET SET AFTER ARGPARSE
train_dataset = None
val_dataset = None


device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Change to 
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()


    

# ========================================================
# ==================   OPTUNA OBJECTIVE   =================
# ========================================================
def objective(trial):
    global train_dataset, val_dataset

    # ----- Hyperparameters -----
    batchnorm = trial.suggest_categorical("batchnorm", [True, False])

    dropout_on = trial.suggest_categorical("dropout_on", [True, False])
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.6) if dropout_on else 0.0

    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    valid_configs = [
        (4, 8), (4, 16), (4, 32), (4, 64),
        (8, 8), (8, 16), (8, 32), (8, 64),
        (16, 8), (16, 16), (16, 32),
        (32, 8), (32, 16),
    ]

    config_idx = trial.suggest_int(
        "config_idx",
        0,
        len(valid_configs) - 1
    )

    batch_size, base_filters = valid_configs[config_idx]

    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("base_filters", base_filters)
    
    epochs = trial.suggest_int("epochs", 50, 200, log=False)


    # ----- Data loaders -----
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ----- Model -----
    model = UNet(
        in_channels=2,
        out_channels=1,
        int_filters=base_filters,
        batchnorm=batchnorm,
        dropout=dropout_rate
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    

    use_amp = device.type in ["cuda", "mps"]

    scaler = GradScaler(
        enabled=(device.type == "cuda")
    )

    amp_dtype = (
        torch.float16 if device.type == "cuda"
        else torch.bfloat16
    )


    best_val_loss = float("inf")



    for epoch in range(epochs):

        # -------- Training --------
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp):
                preds = model(X)
                loss = (0.5 * l1_loss(preds, y)+ 0.5 * mse_loss(preds, y))

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()


        # -------- Validation --------
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                with autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp):
                    preds = model(X)
                    loss = (0.5 * l1_loss(preds, y)+ 0.5 * mse_loss(preds, y))

                val_loss += loss.item() * X.size(0)

        val_loss /= len(val_loader.dataset)

        # Save best model *for this trial*
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    
    # ======================================================
    #  CLEANUP BLOCK — CRITICAL FOR MPS!
    # ======================================================
    del model, optimizer, train_loader, val_loader, scaler
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    
    
    return best_val_loss


# ========================================================
# ============    OPTUNA STUDY SETUP     =================
# ========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNet HPO with Optuna.")

    parser.add_argument(
        "--storage", type=str,
        default=None,
        help="Optuna storage URL. If not given, derived from projectDir."
    )

    parser.add_argument(
        "--n_trials", type=int, default=100,
        help="Number of trials to run"
    )

    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Number of parallel jobs"
    )

    parser.add_argument(
        "--projectDir", type=str,
        default="/Users/rfk471/Dropbox/elevation-canada",
        help="Base project directory"
    )

    parser.add_argument(
        "--experiment", type=str,
        default="EXP1",
        help="Experiment name"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build paths based on projectDir + experiment
    # ------------------------------------------------------------------
    dataDir = os.path.join(args.projectDir, f"data/interim/{args.experiment}")

    # Load datasets
    X_train_hpo = pd.read_pickle(os.path.join(dataDir, "X_train_hpo.pkl"))
    y_train_hpo = pd.read_pickle(os.path.join(dataDir, "y_train_hpo.pkl"))
    X_val_hpo   = pd.read_pickle(os.path.join(dataDir, "X_val_hpo.pkl"))
    y_val_hpo   = pd.read_pickle(os.path.join(dataDir, "y_val_hpo.pkl"))

    # Set global datasets
    train_dataset = ImageDataset(X_train_hpo, y_train_hpo)
    val_dataset   = ImageDataset(X_val_hpo, y_val_hpo)

    # ------------------------------------------------------------------
    # Storage path
    # ------------------------------------------------------------------
    storage = (
        args.storage
        if args.storage is not None
        else f"sqlite:///{os.path.join(args.projectDir, f'models/optuna_hpo_{args.experiment}.db')}"
    )

    # ------------------------------------------------------------------
    # Run Optuna
    # ------------------------------------------------------------------
    study = optuna.create_study(
        direction="minimize",
        study_name=f"udem_hpo_{args.experiment}",
        storage=storage,
        load_if_exists=True
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs
    )


    print("Best trial:", study.best_trial.params)

    # Output pickle file depends on experiment
    out_pkl = os.path.join(
        args.projectDir,
        f"models/best-hpo-{args.experiment}.pkl"
    )

    best_params = study.best_trial.params.copy()
    best_params["batch_size"] = study.best_trial.user_attrs["batch_size"]
    best_params["base_filters"] = study.best_trial.user_attrs["base_filters"]

    with open(out_pkl, "wb") as f:
        pickle.dump(best_params, f)

