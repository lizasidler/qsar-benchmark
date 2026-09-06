import numpy as np
import pandas as pd
import duckdb

import rdkit.Chem as Chem
from rdkit.Chem import rdFingerprintGenerator as fp
from rdkit.Chem import Descriptors as desc
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MolSurf
from rdkit.Chem import Crippen 


with open('fetch_db_data.sql', 'r') as f:
	sql_comm = f.read()

con = duckdb.connect()

con.execute("LOAD sqlite;")
con.execute("ATTACH 'chembl_37.db' AS chembl (TYPE sqlite);")

df = con.execute(sql_comm).df()

fpgen = fp.GetMorganGenerator(radius=2, fpSize=2048)

df['mol'] = df['canonical_smiles'].apply(Chem.MolFromSmiles)
df['fp'] = df['mol'].apply(fpgen.GetCountFingerprintAsNumPy) # train
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


fp = np.vstack(df['fp'].values)[valid_mol]
np.savetxt('fp_only.dat', fp)

activity = np.vstack(df['av_act'])[valid_mol]
np.savetxt('activity.dat', activity)

mol_desc = mol_desc[valid_mol]
np.savetxt('mol_desc.dat', mol_desc)

mol_rep = np.hstack([fp, mol_desc])
np.savetxt('full_mol_rep.dat', mol_rep)



