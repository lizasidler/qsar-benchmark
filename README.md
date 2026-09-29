# qsar-benchmark

This project investigates how different machine learning (ML) approaches perform for quantitative structure–activity
relationship (QSAR) prediction. Different molecular representations are also considered across selected models.

The project benchmarks classical ML models, multilayer perceptrons (MLPs), and graph neural networks (GNNs) for predicting molecular activity. Random Forest and XGBoost are additionally evaluated on a binary classification task to predict whether molecules are active or inactive.

## Key Results

* Both Random Forest and XGBoost achieved an r2 score of 0.67 with Morgan fingerprints.
* `GatedGraphConv` achieved the highest performance among the evaluated GNN architectures with an r2 score of 0.66.
* Among the evaluated configurations, the MLP achieved the highest r2 score (0.73) using Morgan fingerprints + molecular descriptors.

## Datasets & Molecular Representations

* **Database:** ChEMBL
* **Target:** D(2) dopamine receptor, 'CHEMBL217'
* **Dataset Size:** 8200 candidates. All datasets were randomly divided into 90% training and 10% validation data. All reported metrics were calculated on the validation set.

We benchmarked different ML across three distinct molecular data representations: 
* Morgan fingerprints (FP)
* Molecular descriptors (MD)
* Molecular graphs: Node features encode atom types and/or atomic descriptors; optionally, edge features capture bond order

## Models

* Classical ML (Random Forest, XGBoost)
* MLP
* GNN architectures (`SAGEConv`, `GINConv`, `GraphConv`, //`GatedGraphConv`, `ResGatedGraphConv`, `GATConv`, `NNConv`, `GINE`, `Attention`)

## Notebooks & Benchmark Results

Each notebook contains an end-to-end workflow covering data loading, model initialization, training, loss visualization, and metric evaluation (R2 score, root mean squared error (RMSE), mean absolute error (MAE)).

Hyperparameters (such as layer depth, hidden dimensions, `n_estimators`, and `max_depth`) were tuned across models to provide the best score.  All values are approximate, rerun with different seeds might change them but not drastically. The rerun might change R2  error by maximums value of 0.05. The benchmark results presented here are all for validation dataset.

* `random_forest_xgboost_classifiers/` - The Random Forest and XGBoost classifiers trained on Morgan fingerprints.

| Model | R2 score |
| :---           | :---    |
| Random Forest  | 0.80    |
| XGBoost        | 0.82    |
  
* `random_forest_xgboost_regression/` - The Random Forest and XGBoost regression models trained on Morgan fingerprints (FP), molecular descriptors (MD), and their combination (FP+MD).

| Model | Dataset | R2 score |
| :---           | :---        | :---    |
| Random Forest  | FP          | 0.67    |
| Random Forest  | MD          | 0.57    |
| Random Forest  | FP + MD     | 0.62    |
| XGBoost        | FP          | 0.67    |
| XGBoost        | MD          | 0.51    |
| XGBoost        | FP + MD     | 0.63    |

* `MLP_regression/` - The custom MLP regression models trained on Morgan fingerprints (MF), molecular descriptors (MD), and their combination (FP+MD).

| Model | Dataset | R2 score | RMSE | MSA |
| :---  | :---        | :---    |:---    | :---    |
| MLP1  | FP          | 0.68    | 0.56 | 0.41 |
| MLP1  | MD          | 0.50    | 0.71 | 0.53 |
| MLP1  | FP + MD     | 0.70    | 0.55 | 0.42 |
| MLP2  | FP + MD     | 0.73    | 0.52 | 0.39 |

* `GNN_regression/` - The custom GNN regression models trained on the graph dataset with atom-types node features. Different types of convolutional layers are benchmarked.

| Model(layers)      | R2 score | RMSE | MSA |
| :---               | :---    |:---    | :---    |
| SAGEConv           | 0.50    | 0.71 | 0.54 |
| GINConv            | 0.44    | 0.74 | 0.55 |
| GraphConv          | 0.52    | 0.69 | 0.53 |
| GatedGraphConv     | 0.64    | 0.6 | 0.46 |
| ResGatedGraphConv  | 0.54    | 0.67 | 0.51 |
| GATConv            | 0.52    | 0.70 | 0.53 |


* `Gated_GNN_regression/` - The custom GNN regression model with GatedGraphConv layers trained on the graph dataset with atom-types and other atom specific (hybridization type, charge, H-count...) node features.

Val R2 score:
0.66

* `GNN_with_edges_regression/` - The custom GNN regression model trained on the graph dataset with atom-types and other atom specific (hybridization type, charge, H-count...) node features and bond-type edge features. Different types of convolutional layers are benchmarked.

| Model(layers)     | R2 score | RMSE | MSA |
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
* **Deep Learning:** MLP were built with `torch`.
* **GNN:** All GNN architectures were constructed using `torch` and `torch_geometric`.
