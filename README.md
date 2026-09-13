# qsar-benchmark

Benchmarking classical ML models (Random Forest, XGBoost, MLP) and GNN architectures (`SAGEConv`, `GINConv`, `GraphConv`, `GatedGraphConv`, `ResGatedGraphConv`, `GATConv`, `NNConv`, `GINE`, `Attention`) for quantitative structure-activity relationship (QSAR) prediction. Additionally, Random Forest, XGBoost were benchmarked for the classification task. All approaches were evaluated on a regression task predicting molecular activity. Additionally, Random Forest and XGBoost were evaluated on a classification task.

### Key Results (Regression)

* Both Random Forest and XGBoost achieved an r2 score of 0.67 with Morgan fingerprints.
* `GatedGraphConv` achieved the highest performance among the evaluated GNN architectures with an r2 score of 0.66.
* The MLP provided the highest overall r2 score of 0.72 using Morgan fingerprints and molecular descriptors together.

## Datasets & Molecular Representations

* **Database:** ChEMBL
* **Target:** D(2) dopamine receptor, 'CHEMBL217'
* **Dataset Size:** 8200 candidates (90% for training, 10% to validate)

We benchmarked different ML across three distinct molecular data representations: 
* Morgan fingerprints (FP)
* Molecular descriptors (MD)
* Molecular graphs: Node features encode atom types and/or atomic descriptors; optionally, edge features capture bond order

## Repository Structure

* `notebooks/` — Jupyter notebooks containing all benchmarking experiments and workflows.
* `models/` — Custom PyTorch and PyTorch Geometric (PyG) NN architectures.
* `DB/` — Data retrieval and processing scripts for building datasets from ChEMBL. Raw dataset are too large to store in Git.

## Notebooks & Benchmark results (R2 scores):

Each notebook contains an end-to-end workflow covering data loading, model initialization, training, loss visualization, and metric evaluation (R2 score, mean squared error, mean absolute error).

Hyperparameters (such as layer depth, hidden dimensions, `n_estimators`, and `max_depth`) were tuned across models to provide the best score.

* `random_forest_xgboost_classifiers/` - The Random Forest and XGBoost classifiers trained on Morgan fingerprints.

| Model | Val R2 score |
| :---           | :---    |
| Random Forest  | 0.80    |
| XGBoost        | 0.82    |
  
* `random_forest_xgboost_regression/` - The Random Forest and XGBoost regression models trained on Morgan fingerprints (FP), molecular descriptors (MD), and their combination (FP+MD).

| Model | Dataset | Val R2 score |
| :---           | :---        | :---    |         
| Random Forest  | FP          | 0.67    |
| Random Forest  | MD          | 0.57    |
| Random Forest  | FP + MD     | 0.62    |
| XGBoost        | FP          | 0.67    |
| XGBoost        | MD          | 0.51    |
| XGBoost        | FP + MD     | 0.63    |

* `MLP_regression/` - The custom MLP regression models trained on Morgan fingerprints (MF), molecular descriptors (MD), and their combination (FP+MD).

| Model | Dataset |Val R2 score |
| :---  | :---        | :---    |         
| MLP1  | FP          | 0.66    |
| MLP1  | MD          | 0.50    |
| MLP1  | FP + MD     | 0.70    |
| MLP2  | FP + MD     | 0.73    |

* `GNN_regression/` - The custom GNN regression models trained on the graph dataset with atom-types node features. Different types of convolutional layers are benchmarked.

| Model(layers)      | Val R2 score |
| :---               | :---    |         
| SAGEConv           | 0.42    |
| GINConv            | 0.46    |
| GraphConv          | 0.52    |
| GatedGraphConv     | 0.64    |
| ResGatedGraphConv  | 0.54    |
| GATConv            | 0.52    |


* `Gated_GNN_regression/` - The custom GNN regression model with GatedGraphConv layers trained on the graph dataset with atom-types and other atom specific (hybridization type, charge, H-count...) node features.

Val R2 score:
0.66

* `GNN_with_edges_regression/` - The custom GNN regression model trained on the graph dataset with atom-types and other atom specific (hybridization type, charge, H-count...) node features and bond-type edge features. Different types of convolutional layers are benchmarked.

| Model(layers)     | Val R2 score |
| :---              | :---    |         
| GatedEdgeConv     | 0.6     |
| NNConv            | 0.56    |
| GINE              | 0.63    |
| Attention         | 0.6    |


## Dependencies & Libraries

* **Classical ML:** Random Forest models were trained using `scikit-learn`; XGBoost models were trained using `xgboost`.
* **Deep Learning:** MLP were built with `torch`.
* **GNN:** All GNN architectures were constructed using `torch` and `torch_geometric`.
