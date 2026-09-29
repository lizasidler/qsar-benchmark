# qsar-benchmark

This project investigates how different machine learning (ML) approaches perform for quantitative structure–activity
relationship (QSAR) prediction. Different molecular representations are also considered across selected models.

The project benchmarks classical ML models, multilayer perceptrons (MLPs), and graph neural networks (GNNs) for predicting molecular activity. Random Forest and XGBoost are additionally evaluated on a binary classification task to predict whether molecules are active or inactive.

## Key Results

* Both Random Forest and XGBoost regression models achieved an r2 score of 0.67 with Morgan fingerprints.
* `GatedGraphConv` achieved the highest r2 score among the evaluated GNN architectures (0.66) when using atom-type and additional atom-level features.
* Among the evaluated configurations, the MLP achieved the highest r2 score (0.73) using Morgan fingerprints + molecular descriptors.

## Datasets & Molecular Representations

* **Database:** ChEMBL
* **Target:** D(2) dopamine receptor, 'CHEMBL217'
* **Dataset Size:** 8,200 molecules. All datasets were randomly divided into 90% training and 10% validation data. All reported metrics were calculated on the validation set.

We benchmarked different ML approaches across three distinct molecular representations:
* Morgan fingerprints (FP)
* Molecular descriptors (MD)
* Molecular graphs: Node features encode atom types and/or atomic descriptors; optionally, edge features capture bond order

## Models

* Classical ML (Random Forest, XGBoost)
* MLP
* GNN architectures (`SAGEConv`, `GINConv`, `GraphConv`, `GatedGraphConv`, `ResGatedGraphConv`, `GATConv`, `NNConv`, `GINE`, `Attention`)

## Notebooks & Benchmark Results

Notebooks contain an end-to-end workflow covering data loading, model initialisation, training, loss-versus-epoch plots, and evaluation using r2 score, root mean squared error (RMSE), and mean absolute error (MAE).

Hyperparameters such as layer depth, hidden dimensions, `n_estimators`, and `max_depth` were tuned across models. Reported results are from the validation set; due to random initialisation and splitting, exact values may vary between runs.

* `random_forest_xgboost_classifiers/` - The Random Forest and XGBoost classifiers trained on Morgan fingerprints.

| Model | Accuracy |
| :---           | :---    |
| Random Forest  | 0.80    |
| XGBoost        | 0.82    |
  
* `random_forest_xgboost_regression/` - The Random Forest and XGBoost regression models trained on Morgan fingerprints (FP), molecular descriptors (MD), and their combination (FP+MD).

| Model | Dataset | r2 score |
| :---           | :---        | :---    |
| Random Forest  | FP          | 0.67    |
| Random Forest  | MD          | 0.57    |
| Random Forest  | FP + MD     | 0.62    |
| XGBoost        | FP          | 0.67    |
| XGBoost        | MD          | 0.51    |
| XGBoost        | FP + MD     | 0.63    |

* `MLP_regression/` - The custom MLP regression models trained on Morgan fingerprints (FP), molecular descriptors (MD), and their combination (FP+MD).

| Model | Dataset | r2 score | RMSE | MAE |
| :---  | :---        | :---    |:---    | :---    |
| MLP1  | FP          | 0.68    | 0.56 | 0.41 |
| MLP1  | MD          | 0.50    | 0.71 | 0.53 |
| MLP1  | FP + MD     | 0.70    | 0.55 | 0.42 |
| MLP2  | FP + MD     | 0.73    | 0.52 | 0.39 |

* `GNN_regression/` - The custom GNN regression models trained on the graph dataset with atom-type node features. Different types of convolutional layers are benchmarked.

| Model(layers)      | r2 score | RMSE | MAE |
| :---               | :---    |:---    | :---    |
| SAGEConv           | 0.50    | 0.71 | 0.54 |
| GINConv            | 0.44    | 0.74 | 0.55 |
| GraphConv          | 0.52    | 0.69 | 0.53 |
| GatedGraphConv     | 0.64    | 0.6 | 0.46 |
| ResGatedGraphConv  | 0.54    | 0.67 | 0.51 |
| GATConv            | 0.52    | 0.70 | 0.53 |


* `Gated_GNN_regression/` - The custom GNN regression model with GatedGraphConv layers trained on the graph dataset with atom-type and other atom specific (hybridization type, charge, H-count...) node features.

| r2 score | RMSE | MAE |
| :---    |:---    | :---    |
| 0.66     | 0.58 | 0.42 |

* `GNN_with_edges_regression/` - The custom GNN regression model trained on the graph dataset with atom-type and other atom specific (hybridization type, charge, H-count...) node features and bond-type edge features. Different types of convolutional layers are benchmarked.

| Model(layers)     | r2 score | RMSE | MAE |
| :---              | :---    |:---    | :---    |
| GatedEdgeConv     | 0.64     | 0.6 | 0.44 |
| NNConv            | 0.56    | 0.66 | 0.5 |
| GINE              | 0.63    | 0.61 | 0.45 |
| Attention         | 0.6    | 0.63 | 0.47 |

## Repository Structure

* `notebooks/` — Jupyter notebooks containing all benchmarking experiments and workflows.
* `models/` — Custom PyTorch and PyTorch Geometric (PyG) NN architectures.
* `DB/` — Data retrieval and processing scripts for building datasets from ChEMBL. Raw datasets are too large to store in Git.

## Dependencies & Libraries

* **Classical ML:** Random Forest models were trained using `scikit-learn`; XGBoost models were trained using `xgboost`.
* **Deep Learning:** MLPs were built with `torch`.
* **GNN:** All GNN architectures were constructed using `torch` and `torch_geometric`.
