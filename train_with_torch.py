import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import duckdb

import torch
import torch_fp_nn
from torch.utils.data import TensorDataset, DataLoader

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

fpgen = fp.GetMorganGenerator(radius=2, fpSize=2048)

df['mol'] = df['canonical_smiles'].apply(Chem.MolFromSmiles)
df['fp'] = df['mol'].apply(fpgen.GetCountFingerprintAsNumPy) # train
df['max_ch'] = df['mol'].apply(desc.MaxPartialCharge)
df['min_ch'] = df['mol'].apply(desc.MinPartialCharge)
df['mol_w'] = df['mol'].apply(desc.MolWt)
df['log_p'] = df['mol'].apply(Crippen.MolLogP)
df['tpsa'] = df['mol'].apply(MolSurf.TPSA)
#df['sasa'] = df['mol'].apply(MolSurf.LabuteASA)

vsa_names = [name for name, _ in desc._descList 
             if name.startswith(('PEOE_VSA', 'SlogP_VSA'))]

calculator = MoleculeDescriptors.MolecularDescriptorCalculator(vsa_names)
vsa_features = np.array([calculator.CalcDescriptors(m) for m in df['mol']])

mol_desc = np.hstack([df[['max_ch', 'min_ch', 'mol_w', 'log_p', 'tpsa']].values, vsa_features])
valid_mol = ~np.isnan(mol_desc).any(axis=1)

mol_desc = mol_desc[valid_mol]
activity = np.vstack(df['av_act'])[valid_mol]
fp = np.vstack(df['fp'].values)[valid_mol]
indices = np.arange(len(activity))

ind_train, ind_val = train_test_split(
    indices, 
    test_size=0.10,      # 10% held out for evaluation (90% for training)
    random_state=42      # Ensures reproducible splitting
)
act_train, act_val = activity[ind_train], activity[ind_val]
mdesc_train, mdesc_val = mol_desc[ind_train], mol_desc[ind_val]

scaler = StandardScaler()

# Fit on train, transform both train and val
mdesc_train = scaler.fit_transform(mdesc_train)
mdesc_val = scaler.transform(mdesc_val)


fp = np.log1p(fp)
selector = VarianceThreshold(threshold=0.005)
fp_filtr = selector.fit_transform(fp)

fp_train, fp_val = fp_filtr[ind_train], fp_filtr[ind_val]

#mol_rep_train = np.hstack([fp_train, mdesc_train])
#mol_rep_val = np.hstack([fp_val, mdesc_val])

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

#dataset_train = TensorDataset(torch.tensor(mol_rep_train, dtype=torch.float32), torch.tensor(act_train, dtype=torch.float32))
#dataset_val = TensorDataset(torch.tensor(mol_rep_val, dtype=torch.float32), torch.tensor(act_val, dtype=torch.float32))

dataset_train = TensorDataset(torch.tensor(fp_train, dtype=torch.float32), torch.tensor(mdesc_train, dtype=torch.float32), torch.tensor(act_train, dtype=torch.float32))
dataset_val = TensorDataset(torch.tensor(fp_val, dtype=torch.float32), torch.tensor(mdesc_val, dtype=torch.float32), torch.tensor(act_val, dtype=torch.float32))


# 2. Wrap in a DataLoader
train_loader = DataLoader(
    dataset_train, 
    batch_size=100,   
    shuffle=True   
)

val_loader = DataLoader(
    dataset_val, 
    batch_size=len(dataset_val),   
    shuffle=True     
)

epochs = 100
print(fp_train.shape[1], mdesc_train.shape[1])
#model = torch_fp_nn.MLP(mol_rep_train.shape[1], 16, 1).to(device)
model = torch_fp_nn.MLP(fp_train.shape[1], mdesc_train.shape[1], 1).to(device)
model.fit(train_loader, val_loader, device, epochs=epochs)


plt.plot(model.train_loss, label='train')
plt.plot(model.val_loss, label='val')

plt.legend()

plt.xticks(range(0, epochs, 10))

plt.savefig('nn_loss.png')
plt.show()

model.eval()
pred = model(torch.tensor(fp_val, dtype=torch.float32).to(device), torch.tensor(mdesc_val, dtype=torch.float32).to(device))
pred = pred.cpu().detach().numpy()

r2 = r2_score(act_val, pred)
rmse = np.sqrt(mean_squared_error(act_val, pred))
mae = mean_absolute_error(act_val, pred)

print(f"R² Score: {r2:.3f}")
print(f"RMSE:     ±{rmse:.3f}")
print(f"MAE:      ±{mae:.3f}")

