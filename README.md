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
* Morgan fingerprints (FP)
* Molecular descriptors (MD)
* Molecular graphs (node features depend on atom types and/or atomic descriptors + with or without edge features describing bond order )

## Structure:

* We provide JupiterNotebooks in the folder `notebooks` which contain all of the benchmarks. 
* The custom NN designed with PyTorch and pytorch_geometric are collected in the folder `models`.
* In the folder `DB`, all the scripts used to construct datasets from the ChEMBL database are stored. The datasets are too large to store in git.

### Notebooks:

Each notebook for each model contains the end-to-end workflow with data loading, model init, model training, loss plotting, and metrics evaluations (R2 score, mean squared error, mean absolute error)

ML pipeline with:
* The Random Forest and XGBoost classifiers trained on Morgan fingerprints in `random_forest_xgboost_classifiers`

| Model | Val R2 score |
| :---           | :---    |
| Random Forest  | 0.80    |
| XGBoost        | 0.82    |
  
* The Random Forest and XGBoost regression models trained on Morgan fingerprints, molecular descriptors, and their combo in `random_forest_xgboost_regression`. 

| Model | Dataset | Val R2 score |
| :---           | :---        | :---    |         
| Random Forest  | FP          | 0.67    |
| Random Forest  | MD          | 0.57    |
| Random Forest  | FP + MD     | 0.62    |
| XGBoost        | FP          | 0.67    |
| XGBoost        | MD          | 0.51    |
| XGBoost        | FP + MD     | 0.63    |

* The custom MLP regression models trained on Morgan fingerprints, molecular descriptors, and their combo in `MLP_regression`.

| Model | Dataset |Val R2 score |
| :---  | :---        | :---    |         
| MLP1  | FP          | 0.66    |
| MLP1  | MD          | 0.50    |
| MLP1  | FP + MD     | 0.70    |
| MLP2  | FP + MD     | 0.73    |

* The custom GNN regression models trained on graphs datasets with atom-types node features in `GNN_regression`. Different convolutional layers are benchmarked.

| Model(layers)      | Val R2 score |
| :---               | :---    |         
| SAGEConv           | 0.24    |
| GINConv            | 0.23    |
| GraphConv          | 0.47    |
| GatedGraphConv     | 0.62    |
| ResGatedGraphConv  | 0.51    |
| GATConv            | 0.40    |


* The custom GNN regression model with GatedGraphConv layers with atom-types and other atom specific (hybridization type, charge, H-count...) node features in `Gated_GNN_regression`.

Val R2 score:
0.66

* The custom GNN regression model with atom-types and other atom specific (hybridization type, charge, H-count...) node features and bond-type edge features in `GNN_with_edges_regression`. Different convolutional layers are benchmarked.

| Model(layers)     | Val R2 score |
| :---              | :---    |         
| GatedEdgeConv     | 0.6     |
| NNConv            | 0.5     |
| GINE              | 0.61    |
| Attention         | 0.55    |


## Python libraries:
* Random Forest and XGboost were trained using scikit-learn and XGBoost, respectively;
* MLP was implemented using PyTorch;
* GNN were constructed using PyTorch with torch_geometric

