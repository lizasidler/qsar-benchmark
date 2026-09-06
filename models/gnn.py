import torch
import torch.nn as nn 
import torch_geometric.nn as gnn 
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def analysis(self, train_loader, val_loader, device):

	predictions = []
	measured = []

	with torch.no_grad():
	    for batch in train_loader:
	        batch = batch.to(device)  # Move each mini-batch to the GPU/MPS/CPU individually
	        pred = self(batch)
	        predictions.append(pred)
	        measured.append(batch.y)

	all_preds = torch.cat(predictions, dim=0)
	all_preds = all_preds.detach().cpu().numpy().flatten()

	all_measured = torch.cat(measured, dim=0)
	all_measured = all_measured.detach().cpu().numpy().flatten()

	r2 = r2_score(all_measured, all_preds)
	rmse = np.sqrt(mean_squared_error(all_measured, all_preds))
	mae = mean_absolute_error(all_measured, all_preds)

	print(f"R² Train Score: {r2:.3f}")
	print(f"Train RMSE:     ±{rmse:.3f}")
	print(f"Train MAE:      ±{mae:.3f}")


	predictions = []
	measured = []

	with torch.no_grad():
	    for batch in val_loader:
	        batch = batch.to(device)  # Move each mini-batch to the GPU/MPS/CPU individually
	        pred = self(batch)
	        predictions.append(pred)
	        measured.append(batch.y)

	all_preds = torch.cat(predictions, dim=0)
	all_preds = all_preds.detach().cpu().numpy().flatten()

	all_measured = torch.cat(measured, dim=0)
	all_measured = all_measured.detach().cpu().numpy().flatten()

	r2 = r2_score(all_measured, all_preds)
	rmse = np.sqrt(mean_squared_error(all_measured, all_preds))
	mae = mean_absolute_error(all_measured, all_preds)

	print(f"R² Val Score: {r2:.3f}")
	print(f"Val RMSE:     ±{rmse:.3f}")
	print(f"Val MAE:      ±{mae:.3f}")


def fit(self, train_loader, val_loader, device, epochs=100):

	self.train()
	running_train_loss = 0.0
	running_val_loss = 0.0


	for t in tqdm(range(epochs)):

		for batch in train_loader:
			batch = batch.to(device)
			self.optimizer.zero_grad()
			pred = self(batch)
			train_y = batch.y
			loss = self.loss_fn(pred, train_y)
			loss.backward()
			self.optimizer.step()

			running_train_loss += loss.item() * batch.x.size(0)

		running_train_loss /= len(train_loader.dataset)
		self.train_loss.append(running_train_loss)


		self.eval()

		with torch.no_grad():
		    for batch in val_loader:
		    	batch = batch.to(device)

		    	pred = self(batch)
		    	loss = self.loss_fn(pred, batch.y)

		    	running_val_loss += loss.item() * batch.x.size(0)

		running_val_loss /= len(val_loader.dataset)
		self.val_loss.append(running_val_loss)


def ext_forward(self, batch):
	x, edge_index, _batch  = batch.x, batch.edge_index, batch.batch
	
	x = self.embed_at_types(x.long())
	x0 = self.conv1(x, edge_index)
	x1 = self.conv2(x0, edge_index)

	x_all = torch.cat([x, x1], dim=-1)
	x_mean = gnn.global_mean_pool(x_all, _batch)
	x_max = gnn.global_max_pool(x_all, _batch)
	x_add = gnn.global_add_pool(x_all, _batch)

	x = torch.cat([x_mean, x_max, x_add], dim=-1)

	x = self.layer_gnn_h(x)
	out = self.layer_h_out(x)

	return out

class SAGEC(nn.Module):

	fit = fit
	analysis = analysis
	forward = ext_forward

	def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):

		super().__init__()
		self.embed_at_types = nn.Embedding(120, input_dim)
		self.conv1 = gnn.Sequential('x, edge_index', [
			(gnn.GCNConv(input_dim, hidden_dim_1), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_1), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.conv2 = gnn.Sequential('x, edge_index', [
			(gnn.GCNConv(hidden_dim_1, hidden_dim_2), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim_2), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])


		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim_2+input_dim)*3, hidden_dim_3), nn.BatchNorm1d(hidden_dim_3), nn.SiLU())
		self.layer_h_out = nn.Linear(hidden_dim_3, output_dim)


		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []



class GINConv(nn.Module):

	fit = fit
	analysis = analysis
	forward = ext_forward

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
		


class GraphConv(nn.Module):

	fit = fit
	analysis = analysis
	forward = ext_forward

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


class GatedGraphConv(nn.Module):

	fit = fit
	analysis = analysis

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



class ResGatedGraphConv(nn.Module):

	fit = fit
	analysis = analysis
	forward = ext_forward

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




class GATConv(nn.Module):

	fit = fit
	analysis = analysis
	forward = ext_forward

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




