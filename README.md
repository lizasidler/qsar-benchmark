# qsar-benchmark
Benchmarking classical ML (Random Forest, XGBoost, MLP) and GNN approaches (SAGEC, GINConv, GraphConv, GatedGraphConv, ResGatedGraphConv, GATConv, NNConv, GINE, Attention from torch_geometric) for molecular quantitive structure-activity relationship prediction. All approaches were benchmarked for regression task. Additionally, 

GatedGraphConv provided highest score 

4 different ML apploaches were investigated:

-- for both regression and classification, Random Forest and XGboost were trained (using Ski and XGBoost Python librarier)] 
-- for regression, simple MLP NN was constructed and trained
-- for regression, GNN was constructed and different models were testes (...) (using PyTorch and PyTorch Geometric)



## Datasets:

* ChEMBL database
* Target: D(2) dopamine receptor, 'CHEMBL217'
* Activities of 8200 candidates (90% for training, 10% to validate)

We benchmarked different ML approaches in combination with the following molecular data representations: 
* Morgan fingerprints
* Molecular descriptors
* Molecular graphs (node features depend on atom types and/or atomic descriptors + with or without edge features describing bond order )

## Notebooks:


The scores from benchmarking are collected in table : 

