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

df['max_ch'] = df['mol'].apply(desc.MaxPartialCharge)
df['min_ch'] = df['mol'].apply(desc.MinPartialCharge)
df['mol_w'] = df['mol'].apply(desc.MolWt)
df['log_p'] = df['mol'].apply(Crippen.MolLogP)
df['tpsa'] = df['mol'].apply(MolSurf.TPSA)

vsa_names = [name for name, _ in desc._descList 
             if name.startswith(('PEOE_VSA', 'SlogP_VSA'))]

calculator = MoleculeDescriptors.MolecularDescriptorCalculator(vsa_names)
vsa_features = np.array([calculator.CalcDescriptors(m) for m in df['mol']])

mol_desc = np.hstack([df[['max_ch', 'min_ch', 'mol_w', 'log_p', 'tpsa']].values, vsa_features])
valid_mol = ~np.isnan(mol_desc).any(axis=1)

graphs_list = []
at_degree_full = []

for mol, activity in zip(df['mol'], df['av_act']):
    #Chem.AddHs(mol)
    #Chem.RemoveHs(mol)

    
    at_degree_list = []
    mol_feat_list = []

    ComputeGasteigerCharges(mol)

    for atom in mol.GetAtoms():

        p = 1

        at_feat_list = []

        deg = atom.GetDegree() # Number of directly bonded neighbors.
        at_degree_list.append(deg)

        at_nums = atom.GetAtomicNum()
        at_feat_list.append(at_nums)
        at_feat_list.append(deg)
        hydrid = atom.GetHybridization() # Returns RDKit hybridization type (e.g., sp ...)
        at_feat_list.append(int(hydrid))
        p_charge = atom.GetProp("_GasteigerCharge")
        at_feat_list.append(float(p_charge))
        charge = atom.GetFormalCharge() # Formal charge on the atom.
        at_feat_list.append(charge)
        arom = atom.GetIsAromatic() # Boolean indicating aromaticity.
        at_feat_list.append(arom*1.0)
        Hcount = atom.GetTotalNumHs() # Count of connected hydrogens.
        at_feat_list.append(Hcount)

        if np.isnan(at_feat_list).any():
            print('oops',at_feat_list)
            p=0
            break

        mol_feat_list.append(at_feat_list)


    if p==0:
        continue

    mol_feat_tensor = torch.tensor(mol_feat_list, dtype=torch.float32)

    if np.isnan(mol_feat_tensor).any():
            print('oops')

    at_degree_full.append(at_degree_list)
    edge_id_list = []
    edge_type_list = []

    for bond in mol.GetBonds():
        id_1 = bond.GetBeginAtomIdx() #— Source node index.
        id_2 = bond.GetEndAtomIdx() #— Target node index.
        edge_id_list.append([id_1, id_2])
        edge_id_list.append([id_2, id_1])

        edge_type = BOND_MAP[str(bond.GetBondType())]
        edge_type_list.append(edge_type)
        edge_type_list.append(edge_type)
        print(edge_type)

    edge_id_tensor = torch.tensor(edge_id_list, dtype=torch.long).t().contiguous()
    edge_type_tensor = torch.tensor(edge_type_list, dtype=torch.long).t().contiguous()

    graph = Data(x = mol_feat_tensor, edge_index = edge_id_tensor, edge_attr=edge_type_tensor, y = torch.tensor([[activity]],  dtype=torch.float32))
    graph.validate(raise_on_error=True)
    graphs_list.append(graph)

torch.save(graphs_list, 'gnn_dt_feat_edges.pt')

torch.save(graphs_list, 'gnn_dt_feat_with_rings.pt')

graphs_train, graphs_val, at_degree_train, at_degree_val = train_test_split(
    graphs_list, at_degree_full,
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

at_degree_train = [i for mol in at_degree_train for i in mol]
at_degree_val = [i for mol in at_degree_val for i in mol]

at_degree_tensor = torch.tensor(at_degree_train, dtype=torch.long)
at_degree_train_hist = torch.bincount(at_degree_tensor)

#at_degree_val_hist = torch.tensor(np.histogram(at_degree_val, bins=range(0, max(at_degree_val)+1)))

epochs = 100
model = torch_GCN.PNAConv(8, 64, 1, at_degree_train_hist).to(device)
#model = torch_GCN.ResGatedGraphConv(8, 64, 1).to(device)
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

