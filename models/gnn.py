import torch
import torch.nn as nn 
import torch_geometric.nn as gnn 
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from .gnn_base_class import BaseClass


class SAGEC(BaseClass):

	def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):

		super().__init__()
		self.embed_at_types = nn.Embedding(120, input_dim)
		self.conv1 = gnn.Sequential('x, edge_index', [
			(gnn.SAGEConv(input_dim, hidden_dim_1), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_1), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.conv2 = gnn.Sequential('x, edge_index', [
			(gnn.SAGEConv(hidden_dim_1, hidden_dim_2), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_2), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])


		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim_2+input_dim)*3, hidden_dim_3), nn.BatchNorm1d(hidden_dim_3), nn.SiLU())
		self.layer_h_out = nn.Linear(hidden_dim_3, output_dim)


		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []



class GINConv(BaseClass):

	def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):

		super().__init__()
		self.embed_at_types = nn.Embedding(120, input_dim)

		#GCNonv nope

		self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim_1), nn.LayerNorm(hidden_dim_1), nn.SiLU()))
		self.conv2 = gnn.GINConv(nn.Sequential(nn.Linear(hidden_dim_1, hidden_dim_2), nn.LayerNorm(hidden_dim_2), nn.SiLU()))

		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim_2+input_dim)*3, hidden_dim_3), nn.BatchNorm1d(hidden_dim_3), nn.SiLU())
		self.layer_h_out = nn.Linear(hidden_dim_3, output_dim)

		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []
		


class GraphConv(BaseClass):

	def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):

		super().__init__()
		self.embed_at_types = nn.Embedding(120, input_dim)

		self.conv1 = gnn.Sequential('x, edge_index', [
			(gnn.GraphConv(input_dim, hidden_dim_1), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_1), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.conv2 = gnn.Sequential('x, edge_index', [
			(gnn.GraphConv(hidden_dim_1, hidden_dim_2), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_2), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim_2+input_dim)*3, hidden_dim_3), nn.BatchNorm1d(hidden_dim_3), nn.SiLU())
		self.layer_h_out = nn.Linear(hidden_dim_3, output_dim)


		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []


class GatedGraphConv(BaseClass):

	def __init__(self, input_dim, hidden_dim_1, hidden_dim_3, output_dim):

		super().__init__()
		self.embed_at_types = nn.Embedding(120, input_dim)
		self.layer_node_feat = nn.Sequential(nn.Linear(input_dim, hidden_dim_1), nn.LayerNorm(hidden_dim_1), nn.SiLU())

		self.conv1 = gnn.Sequential('x, edge_index', [
			(gnn.GatedGraphConv(hidden_dim_1, 2), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_1), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.conv2 = gnn.Sequential('x, edge_index', [
			(gnn.GatedGraphConv(hidden_dim_1, 2), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_1), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim_1+input_dim)*3, hidden_dim_3), nn.BatchNorm1d(hidden_dim_3), nn.SiLU())
		self.layer_h_out = nn.Linear(hidden_dim_3, output_dim)


		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []

	def forward(self, batch):
		x, edge_index, _batch  = batch.x, batch.edge_index, batch.batch
		
		x = self.embed_at_types(x.long())
		x0 = self.layer_node_feat(x)

		x1 = self.conv1(x0, edge_index)
		x2 = self.conv2(x1, edge_index)

		x_all = torch.cat([x, x2], dim=-1)
		x_mean = gnn.global_mean_pool(x_all, _batch)
		x_max = gnn.global_max_pool(x_all, _batch)
		x_add = gnn.global_add_pool(x_all, _batch)

		x = torch.cat([x_mean, x_max, x_add], dim=-1)

		x = self.layer_gnn_h(x)
		out = self.layer_h_out(x)

		return out



class ResGatedGraphConv(BaseClass):

	def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):

		super().__init__()
		self.embed_at_types = nn.Embedding(120, input_dim)

		self.conv1 = gnn.Sequential('x, edge_index', [
			(gnn.ResGatedGraphConv(input_dim, hidden_dim_1), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_1), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.conv2 = gnn.Sequential('x, edge_index', [
			(gnn.ResGatedGraphConv(hidden_dim_1, hidden_dim_2), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_2), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim_2+input_dim)*3, hidden_dim_3), nn.BatchNorm1d(hidden_dim_3), nn.SiLU())
		self.layer_h_out = nn.Linear(hidden_dim_3, output_dim)


		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []




class GATConv(BaseClass):

	def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):

		super().__init__()
		self.embed_at_types = nn.Embedding(120, input_dim)

		self.conv1 = gnn.Sequential('x, edge_index', [
			(gnn.GATConv(input_dim, hidden_dim_1, heads=2, concat=True), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_1*2), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.conv2 = gnn.Sequential('x, edge_index', [
			(gnn.GATConv(hidden_dim_1*2, hidden_dim_2, heads=6, concat=False), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_2), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim_2+input_dim)*3, hidden_dim_3), nn.BatchNorm1d(hidden_dim_3), nn.SiLU())
		self.layer_h_out = nn.Linear(hidden_dim_3, output_dim)


		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []




