# qsar-benchmark

Benchmarking classical ML (Random Forest, XGBoost, MLP) and GNN approaches (SAGEC, GINConv, GraphConv, GatedGraphConv, ResGatedGraphConv, GATConv, NNConv, GINE, Attention) for molecular quantitive structure-activity relationship prediction. All approaches were benchmarked for regression task. Additionally, 

GatedGraphConv provided the highest score among the explored GNN (0.69).
Random Forest resulted in the 
The MLP provided the highest overall score of 0.72 for 

## Datasets:

* ChEMBL database
* Target: D(2) dopamine receptor, 'CHEMBL217'
* Activities of 8200 candidates (90% for training, 10% to validate)

We benchmarked different ML approaches in combination with the following molecular data representations: 
* Morgan fingerprints
* Molecular descriptors
* Molecular graphs (node features depend on atom types and/or atomic descriptors + with or without edge features describing bond order )

## Structure:

* We provide JupiterNotebooks in the folder `notebooks` which contain all of the benchmarks. 
* The NN designed with PyTorch and torch_geometric are collected in the folder `models`.
* In the folder `DB`, all the scripts used to construct datasets from the ChEMBL database are stored. The datasets are too large to store 


## Python libraries:
* Random Forest and XGboost were trained using scikit-learn and XGBoost, respectively;
* MLP was implemented using PyTorch;
* GNN were constructed using PyTorch with torch_geometric

All scores collected with the presented benchmarking are collected in table : 

