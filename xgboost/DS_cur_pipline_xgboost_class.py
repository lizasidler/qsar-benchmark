import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt

import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
from rdkit.Chem import rdFingerprintGenerator as fp
from rdkit.Chem import Descriptors as desc

from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

import xgboost 

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


fp = np.vstack(df['fp'].values)
activity = df['av_act']>6.5

clf = xgboost.XGBClassifier(n_estimators=50, random_state=42, learning_rate=0.02, max_depth=35, min_child_weight=5)

fp_train, fp_test, act_train, act_test = train_test_split(
    fp, activity, 
    test_size=0.10,      # 10% held out for evaluation (90% for training)
    random_state=42      # Ensures reproducible splitting
)

clf.fit(fp_train, act_train, eval_set=[(fp_test, act_test)], verbose=False)

train_score = clf.score(fp_train, act_train)
test_score = clf.score(fp_test, act_test)

print(f"Train Score: {train_score:.2f}")
print(f"Test Score:  {test_score:.2f}")

#single_tree = clf.estimators_[0]

#plt.figure(figsize=(20, 10))
#plot_tree(single_tree, max_depth=3, filled=True, rounded=True)
#plt.show()

#depths = [tree.get_depth() for tree in clf.estimators_]

#print(depths)