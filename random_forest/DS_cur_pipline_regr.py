import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt

import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
from rdkit.Chem import rdFingerprintGenerator as fp
from rdkit.Chem import Descriptors as desc


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

from sklearn.feature_selection import VarianceThreshold
from rdkit.Chem import MolSurf
from rdkit.Chem import Crippen 
from rdkit.Chem import Descriptors as desc

from rdkit.ML.Descriptors import MoleculeDescriptors

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

fp = np.vstack(df['fp'].values)

selector = VarianceThreshold(threshold=0.01)
fp_filtr = selector.fit_transform(fp)
print(fp_filtr.shape)

mol_rep = np.hstack([fp_filtr, df[['max_ch', 'min_ch', 'mol_w', 'log_p', 'tpsa']].values, vsa_features])
activity = np.vstack(df['av_act'])

clf = RandomForestRegressor(n_estimators=40, random_state=42, max_depth=50, min_samples_leaf=3)

fp_train, fp_test, act_train, act_test = train_test_split(
    mol_rep, activity, 
    test_size=0.10,      # 10% held out for evaluation (90% for training)
    random_state=42      # Ensures reproducible splitting
)

clf.fit(fp_train, act_train)

train_score = clf.score(fp_train, act_train)
test_score = clf.score(fp_test, act_test)

print(f"Train Score: {train_score:.2f}")
print(f"Test Score:  {test_score:.2f}")
