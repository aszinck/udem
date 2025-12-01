import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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

criterion = nn.MSELoss()


    

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
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    base_filters = trial.suggest_categorical("base_filters", [8, 16, 32, 64, 128])

    epochs = trial.suggest_int("epochs", 50, 200, log=False)

    # ----- Data loaders -----
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ----- Model -----
    model = UNet(
        in_channels=3,
        out_channels=1,
        int_filters=base_filters,
        batchnorm=batchnorm,
        dropout=dropout_rate
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)


    #EPOCHS = epochs  # You can lower during development (e.g., 20)
    best_val_loss = float("inf")


    for epoch in range(epochs):

        # -------- Training --------
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        # -------- Validation --------
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                val_loss += criterion(preds, y).item() * X.size(0)

        val_loss /= len(val_loader.dataset)

        # Save best model *for this trial*
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            # Cleanup before pruning too
            del model, optimizer, train_loader, val_loader
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    
    # ======================================================
    #  CLEANUP BLOCK — CRITICAL FOR MPS!
    # ======================================================
    del model
    del optimizer
    del train_loader
    del val_loader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()
    
    
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
        "--normalization", type=str,
        default="raw",
        help="Normalization type (raw, zscore, etc.)"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build paths based on projectDir + normalization
    # ------------------------------------------------------------------
    dataDir = os.path.join(args.projectDir, f"data/interim/normalization-{args.normalization}")

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
        else f"sqlite:///{os.path.join(args.projectDir, f'models/optuna_hpo_{args.normalization}.db')}"
    )

    # ------------------------------------------------------------------
    # Run Optuna
    # ------------------------------------------------------------------
    study = optuna.create_study(
        direction="minimize",
        study_name=f"udem_hpo_{args.normalization}",
        storage=storage,
        load_if_exists=True
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs
    )

    print("Best trial:", study.best_trial.params)

    # Output pickle file depends on normalization
    out_pkl = os.path.join(
        args.projectDir,
        f"models/best-hpo-{args.normalization}.pkl"
    )

    with open(out_pkl, "wb") as f:
        pickle.dump(study.best_trial.params, f)

