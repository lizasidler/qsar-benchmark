import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import duckdb

import torch
from torch_geometric.data import Data

import rdkit.Chem as Chem
from rdkit.Chem import MolSurf
from rdkit.Chem import Crippen 
from rdkit.Chem import Descriptors as desc
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

with open('fetch_db_data.sql', 'r') as f:
	sql_comm = f.read()


con = duckdb.connect()

con.execute("LOAD sqlite;")
con.execute("ATTACH 'chembl_37.db' AS chembl (TYPE sqlite);")

df = con.execute(sql_comm).df()
print(len(df))

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
valid_mol = iter(~np.isnan(mol_desc).any(axis=1))

CHIRAL_TAG_DICT = {
    'CHI_UNSPECIFIED': 0,
    'CHI_TETRAHEDRAL_CW': 1,
    'CHI_TETRAHEDRAL_CCW': 2,
    'CHI_OTHER': 3
}

graphs_list = []

for mol, activity in zip(df['mol'], df['av_act']):

    if not next(valid_mol):
        continue

    mol_feat_list = []
    ComputeGasteigerCharges(mol)

    p = 1

    for atom in mol.GetAtoms():

        at_feat_list = []

        at_nums = atom.GetAtomicNum()
        at_feat_list.append(at_nums)

        deg = atom.GetDegree() # Number of directly bonded neighbors
        at_feat_list.append(deg)

        hydrid = atom.GetHybridization() # Returns RDKit hybridization type (e.g., sp ...)
        at_feat_list.append(int(hydrid))

        p_charge = atom.GetProp("_GasteigerCharge")
        at_feat_list.append(float(p_charge))

        if np.isnan(float(p_charge)):
            p = 0
            break

        charge = atom.GetFormalCharge() # Formal charge on the atom.
        at_feat_list.append(charge)

        arom = atom.GetIsAromatic() # Boolean indicating aromaticity.
        at_feat_list.append(arom*1.0)

        Hcount = atom.GetTotalNumHs() # Count of connected hydrogens.
        at_feat_list.append(Hcount)

        ring = atom.IsInRing() # Count of connected hydrogens.
        at_feat_list.append(ring*1.0)

        ring_5 = atom.IsInRingSize(5) # Count of connected hydrogens.
        at_feat_list.append(ring_5*1.0)

        ring_6 = atom.IsInRingSize(6) # Count of connected hydrogens.
        at_feat_list.append(ring_6*1.0)

        chiral = CHIRAL_TAG_DICT.get(str(atom.GetChiralTag())) # Count of connected hydrogens.
        at_feat_list.append(chiral)

        mol_feat_list.append(at_feat_list)


    if p == 0:
        continue

    mol_feat_tensor = torch.tensor(mol_feat_list, dtype=torch.float32)

    edge_id_list = []
    edge_type_list = []

    for bond in mol.GetBonds():
        id_1 = bond.GetBeginAtomIdx() #— Source node index.
        id_2 = bond.GetEndAtomIdx() #— Target node index.
        edge_id_list.append([id_1, id_2])
        edge_id_list.append([id_2, id_1])


    edge_id_tensor = torch.tensor(edge_id_list, dtype=torch.long).t().contiguous()

    graph = Data(x = mol_feat_tensor, edge_index = edge_id_tensor, y = torch.tensor([[activity]],  dtype=torch.float32))
    graph.validate(raise_on_error=True)
    graphs_list.append(graph)

torch.save(graphs_list, 'graphs.pt')