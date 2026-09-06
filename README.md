# qsar-benchmark
Benchmarking classical ML and GNN approaches (Random Forest, MLP, GCN) for molecular structure-activity relationship prediction



4 different ML apploaches were investigated:

-- for both regression and classification, Random Forest and XGboost were trained (using Ski and XGBoost Python librarier)]
-- for regression, simple MLP NN was constructed and trained
-- for regression, GNN was constructed and different models were testes (...) (using PyTorch and PyTorch Geometric)



datasets:

-- pure Morgan fingerprints only
--molecular descriptors (...)
-- FP + molecular descriptors (combination of the first two sets)

Molecular graphs datasets
-- node represented by atom types only, no edge features
-- node represented by atom types and atomic descriptors, no edge features
-- node represented by atom types and atomic descriptors, edge features of bond order

The scores from benchmarking are collected in table : 

