import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import duckdb

import torch
import torch_NNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
from rdkit.Chem import MolSurf
from rdkit.Chem import Crippen 
from rdkit.Chem import Descriptors as desc
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdFingerprintGenerator as fp
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges


graphs_list = torch.load('gnn_dt_feat_edges.pt', weights_only=False)

#valid_mask ?

graphs_train, graphs_val = train_test_split(
    graphs_list,
    test_size=0.10,      # 10% held out for evaluation (90% for training)
    random_state=42      # Ensures reproducible splitting
)

if torch.backends.mps.is_available():
    device = torch.device("cpu")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")


# 2. Wrap in a DataLoader
train_loader = DataLoader(
    graphs_train, 
    batch_size=300,   
    shuffle=True   
)

val_loader = DataLoader(
    graphs_val, 
    batch_size=300,   
    shuffle=True   
)


epochs = 200
model = torch_NNConv.Attention(64, 1).to(device)
model.fit(train_loader, val_loader, device, epochs=epochs)


plt.plot(model.train_loss, label='train')
plt.plot(model.val_loss, label='val')

plt.legend()

plt.xticks(range(0, epochs, 10))

plt.savefig('nn_loss.png')
plt.show()

model.eval()
predictions = []
measured = []

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)  # Move each mini-batch to the GPU/MPS/CPU individually
        pred = model(batch)
        predictions.append(pred)
        #measured.append((batch.y-4)/4 )
        measured.append(batch.y)



all_preds = torch.cat(predictions, dim=0)
all_preds = all_preds.detach().cpu().numpy().flatten()

all_measured = torch.cat(measured, dim=0)
all_measured = all_measured.detach().cpu().numpy().flatten()

a, b = np.polyfit(all_preds, all_measured, 1)
r2_calibrated = r2_score(all_measured, a * all_preds + b)

print(f"Optimal Rescaled R2: {r2_calibrated:.4f}")

r2 = r2_score(all_measured, all_preds)
rmse = np.sqrt(mean_squared_error(all_measured, all_preds))
mae = mean_absolute_error(all_measured, all_preds)

print(f"R² Score: {r2:.3f}")
print(f"RMSE:     ±{rmse:.3f}")
print(f"MAE:      ±{mae:.3f}")

