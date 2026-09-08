import os
import pickle
import pandas as pd 
import udem
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler 
import argparse


parser = argparse.ArgumentParser(description="Train UNet experiment")
parser.add_argument("--experiment", type=str, default="EXP1", help="Experiment name (e.g. EXP1, EXP2, EXP3)")
parser.add_argument("--projectDir", type=str, default="/home/aszinck/projects/def-ddj/aszinck/elevation-canada", help="Project directory")

args = parser.parse_args()
experiment = args.experiment
projectDir = args.projectDir

best_params_dir = os.path.join(projectDir,f"models/best-hpo-{experiment}.pkl")

with open(best_params_dir, "rb") as input_file:
    best_params = pickle.load(input_file)

dataDir = f"{projectDir}/data/interim/{experiment}"
X_train = pd.read_pickle(f'{dataDir}/X_train.pkl')
y_train = pd.read_pickle(f'{dataDir}/y_train.pkl')
X_val = pd.read_pickle(f'{dataDir}/X_val.pkl')
y_val = pd.read_pickle(f'{dataDir}/y_val.pkl')

train_dataset = udem.ImageDataset(X_train, y_train)
val_dataset   = udem.ImageDataset(X_val, y_val)


device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)

dropout_rate = best_params["dropout_rate"] if best_params["dropout_on"] else 0.0

model = udem.UNet(
        in_channels=2,
        out_channels=1,
        int_filters=best_params["base_filters"],
        batchnorm=best_params["batchnorm"],
        dropout=dropout_rate
    ).to(device)

optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])

l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)

train_losses = []
val_losses = []

EPOCHS = best_params["epochs"]

scale = GradScaler("mps") 

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast(device_type="mps",dtype=torch.bfloat16): 
            preds = model(X) 
            loss = 0.5 * l1_loss(preds, y) + 0.5 * mse_loss(preds, y)
        scale.scale(loss).backward() 
        scale.step(optimizer) 
        scale.update() 
        train_loss += loss.item() * X.size(0)


    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss) 
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            preds = model(X)
            loss = 0.5 * l1_loss(preds, y) + 0.5 * mse_loss(preds, y)
            val_loss += loss.item() * X.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}",flush=True)
    

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{projectDir}/figures/training-{experiment}.png",dpi=300)

losses_path = os.path.join(projectDir, f"models/losses_{experiment}.pkl")

loss_dict = {
    "train_losses": train_losses,
    "val_losses": val_losses
}

with open(losses_path, "wb") as f:
    pickle.dump(loss_dict, f)


os.makedirs(f"{projectDir}/models/", exist_ok=True)
torch.save(model.state_dict(), f"{projectDir}/models/unet_{experiment}.pth")
