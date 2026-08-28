import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import duckdb

import torch
import torch_GCN
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




with open('fetch_db_data.sql', 'r') as f:
	sql_comm = f.read()


con = duckdb.connect()

con.execute("LOAD sqlite;")
con.execute("ATTACH 'chembl_37.db' AS chembl (TYPE sqlite);")

df = con.execute(sql_comm).df()

#mol = Chem.MolFromSmiles(df['canonical_smiles'][0])
#Draw.ShowMol(mol)

df['mol'] = df['canonical_smiles'].apply(Chem.MolFromSmiles)

graphs_list = []

for mol, activity in zip(df['mol'], df['av_act']):
    #Chem.AddHs(mol)
    #Chem.RemoveHs(mol)

    at_id_list = []

    for atom in mol.GetAtoms():
        at_nums = atom.GetAtomicNum()
        at_id_list.append(at_nums)

        #atom.GetDegree() — Number of directly bonded neighbors.
        #atom.GetHybridization() — Returns RDKit hybridization type (e.g., sp 
        #atom.GetFormalCharge() — Formal charge on the atom.
        #atom.GetIsAromatic() — Boolean indicating aromaticity.
        #atom.GetTotalNumHs() — Count of connected hydrogens.

    at_id_tensor = torch.tensor(at_id_list, dtype=torch.float32)
    edge_id_list = []

    for bond in mol.GetBonds():
        id_1 = bond.GetBeginAtomIdx() #— Source node index.
        id_2 = bond.GetEndAtomIdx() #— Target node index.
        edge_id_list.append([id_1, id_2])
        edge_id_list.append([id_2, id_1])

        #bond.GetBondType()

    edge_id_tensor = torch.tensor(edge_id_list, dtype=torch.long).t().contiguous()

    graph = Data(x = at_id_tensor, edge_index = edge_id_tensor, y = torch.tensor([[activity]], dtype=torch.float32))
    graph.validate(raise_on_error=True)
    graphs_list.append(graph)



#valid_mask ?

graphs_train, graphs_val = train_test_split(
    graphs_list, 
    test_size=0.10,      # 10% held out for evaluation (90% for training)
    random_state=42      # Ensures reproducible splitting
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
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

epochs = 15
model = torch_GCN.GCN(32, 64, 1).to(device)
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
        measured.append((batch.y-4)/4 )



all_preds = torch.cat(predictions, dim=0)
all_preds = all_preds.detach().cpu().numpy().flatten()

all_measured = torch.cat(measured, dim=0)
all_measured = all_measured.detach().cpu().numpy().flatten()

r2 = r2_score(all_measured, all_preds)
rmse = np.sqrt(mean_squared_error(all_measured, all_preds))
mae = mean_absolute_error(all_measured, all_preds)

print(f"R² Score: {r2:.3f}")
print(f"RMSE:     ±{rmse:.3f}")
print(f"MAE:      ±{mae:.3f}")

