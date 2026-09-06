import numpy as np
import pandas as pd
import duckdb

import rdkit.Chem as Chem
from rdkit.Chem import rdFingerprintGenerator as fp


with open('fetch_db_data.sql', 'r') as f:
	sql_comm = f.read()


con = duckdb.connect()

con.execute("LOAD sqlite;")
con.execute("ATTACH 'chembl_37.db' AS chembl (TYPE sqlite);")

df = con.execute(sql_comm).df()

fpgen = fp.GetMorganGenerator(radius=2, fpSize=2048)

df['mol'] = df['canonical_smiles'].apply(Chem.MolFromSmiles)
df['fp'] = df['mol'].apply(fpgen.GetCountFingerprintAsNumPy) # train


fp = np.vstack(df['fp'].values)
activity = np.array(df['av_act']>6.5)


np.savetxt('fp.dat', fp)
np.savetxt('activity_class.dat', activity)

