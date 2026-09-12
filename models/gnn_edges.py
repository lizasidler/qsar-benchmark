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
	        batch = batch.to(device)
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
	        batch = batch.to(device)
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


class GatedEdge(gnn.MessagePassing):

	def __init__(self, hidden_dim, num_steps=2):
		super().__init__(aggr='add')

		self.edge_weight = nn.Sequential(nn.Linear(4, 1), nn.Tanh())
		self.gru = nn.GRUCell(hidden_dim, hidden_dim)
		self.num_steps = num_steps

	def forward(self, x, edge_index, edge_attr):

		for _ in range(self.num_steps):

			out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
			x = self.gru(out, x)

		return x

	def message(self, x_j, edge_attr):
	    
	    edge_attr = self.edge_weight(edge_attr)

	    return  x_j * (1.0 + edge_attr)

class GatedEdgeConv(nn.Module):

	analysis = analysis
	fit = fit

	def __init__(self, hidden_dim_1, hidden_dim_2, output_dim):

		super().__init__()
		self.embed_types = nn.Embedding(120, 8)
		self.embed_hybrid = nn.Embedding(24, 4)
		self.embed_charge = nn.Embedding(12, 4)
		self.embed_hcount = nn.Embedding(12, 4)
		self.embed_deg = nn.Embedding(12, 4)
		self.embed_edges = nn.Embedding(6, 4)
		self.embed_chiral = nn.Embedding(12, 4)

		self.p_charge_nn = nn.Linear(1, 4)

		self.layer_node_feat = nn.Sequential(nn.Linear(33, hidden_dim_1), nn.LayerNorm(hidden_dim_1), nn.SiLU())

		self.conv1 = gnn.Sequential('x, edge_index, edge_attr', [
			(GatedEdge(hidden_dim_1), 'x, edge_index, edge_attr -> x'),
			(nn.LayerNorm(hidden_dim_1), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])


		self.conv2 = gnn.Sequential('x, edge_index, edge_attr', [
			(GatedEdge(hidden_dim_1), 'x, edge_index, edge_attr -> x'),
			(nn.LayerNorm(hidden_dim_1), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim_1+33)*3, hidden_dim_2), nn.BatchNorm1d(hidden_dim_2), nn.SiLU())
		self.layer_h_out = nn.Linear(hidden_dim_2, output_dim)


		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []


	def forward(self, batch):
		x, edge_index, edge_attr, _batch  = batch.x, batch.edge_index, batch.edge_attr, batch.batch

		types_emb = self.embed_types(x[:, 0].long())
		deg_emb = self.embed_deg(x[:, 1].long())
		hydrid_emb = self.embed_deg(x[:, 2].long())
		p_charge = x[:, 3].clone().to(torch.float32).reshape(-1, 1)
		charge_emb = self.embed_charge((x[:, 4]+10).long())
		arom = x[:, 5].clone().to(torch.float32).reshape(-1, 1)
		hcount_emb = self.embed_hcount((x[:, 6]).long())
		ring = x[:, 7].clone().to(torch.float32).reshape(-1, 1)
		ring_5 = x[:, 8].clone().to(torch.float32).reshape(-1, 1)
		ring_6 = x[:, 9].clone().to(torch.float32).reshape(-1, 1)
		chiral = self.embed_chiral((x[:, 10]).long())

		edge_attr = self.embed_edges(edge_attr.long())

		x = torch.cat((types_emb, deg_emb, hydrid_emb, p_charge, charge_emb, arom, hcount_emb, ring, ring_5, ring_6, chiral), dim=1)
		x0 = self.layer_node_feat(x)
		x1 = self.conv1(x0, edge_index, edge_attr)
		x2 = self.conv1(x1, edge_index, edge_attr)

		x_all = torch.cat([x, x2], dim=-1)
		x_mean = gnn.global_mean_pool(x_all, _batch)
		x_max = gnn.global_max_pool(x_all, _batch)
		x_add = gnn.global_add_pool(x_all, _batch)

		x = torch.cat([x_mean, x_max, x_add], dim=-1)

		x = self.layer_gnn_h(x)
		out = self.layer_h_out(x)

		return out



class NNConv(nn.Module):

	analysis = analysis
	fit = fit

	def __init__(self, hidden_dim_1, hidden_dim_2, output_dim):

		super().__init__()
		self.embed_types = nn.Embedding(120, 8)
		self.embed_hybrid = nn.Embedding(24, 4)
		self.embed_charge = nn.Embedding(12, 4)
		self.embed_hcount = nn.Embedding(12, 4)
		self.embed_deg = nn.Embedding(12, 4)
		self.embed_edges = nn.Embedding(6, 4)
		self.embed_chiral = nn.Embedding(12, 4)

		self.p_charge_nn = nn.Linear(1, 4)

		self.edge_nn_1 = nn.Sequential(
	    	nn.Linear(4, 32),
	    	nn.ReLU(),
	    	nn.Linear(32, 33 * hidden_dim_1)
				)
		self.conv1 = gnn.Sequential('x, edge_index, edge_attr', [
			(gnn.NNConv(33, hidden_dim_1, self.edge_nn_1, aggr='add'), 'x, edge_index, edge_attr -> x'),
			(nn.LayerNorm(hidden_dim_1), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])


		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim_1+33)*3, hidden_dim_2), nn.BatchNorm1d(hidden_dim_2), nn.SiLU())
		self.layer_h_out = nn.Linear(hidden_dim_2, output_dim)


		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []


	def forward(self, batch):
		x, edge_index, edge_attr, _batch  = batch.x, batch.edge_index, batch.edge_attr, batch.batch

		types_emb = self.embed_types(x[:, 0].long())
		deg_emb = self.embed_deg(x[:, 1].long())
		hydrid_emb = self.embed_deg(x[:, 2].long())
		p_charge = x[:, 3].clone().to(torch.float32).reshape(-1, 1)
		charge_emb = self.embed_charge((x[:, 4]+10).long())
		arom = x[:, 5].clone().to(torch.float32).reshape(-1, 1)
		hcount_emb = self.embed_hcount((x[:, 6]).long())
		ring = x[:, 7].clone().to(torch.float32).reshape(-1, 1)
		ring_5 = x[:, 8].clone().to(torch.float32).reshape(-1, 1)
		ring_6 = x[:, 9].clone().to(torch.float32).reshape(-1, 1)
		chiral = self.embed_chiral((x[:, 10]).long())

		edge_attr = self.embed_edges(edge_attr.long())

		x = torch.cat((types_emb, deg_emb, hydrid_emb, p_charge, charge_emb, arom, hcount_emb, ring, ring_5, ring_6, chiral), dim=1)
		x0 = self.conv1(x, edge_index, edge_attr)
		#x1 = self.conv2(x0, edge_index, edge_attr)

		x_all = torch.cat([x, x0], dim=-1)
		x_mean = gnn.global_mean_pool(x_all, _batch)
		x_max = gnn.global_max_pool(x_all, _batch)
		x_add = gnn.global_add_pool(x_all, _batch)

		x = torch.cat([x_mean, x_max, x_add], dim=-1)

		x = self.layer_gnn_h(x)
		out = self.layer_h_out(x)

		return out



class GINE(nn.Module):

	analysis = analysis
	fit = fit

	def __init__(self, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):

		super().__init__()
		self.embed_types = nn.Embedding(120, 8)
		self.embed_hybrid = nn.Embedding(24, 4)
		self.embed_charge = nn.Embedding(12, 4)
		self.embed_hcount = nn.Embedding(12, 4)
		self.embed_deg = nn.Embedding(12, 4)
		self.embed_edges = nn.Embedding(6, 4)
		self.embed_chiral = nn.Embedding(12, 4)

		self.conv1 = gnn.Sequential('x, edge_index, edge_attr', [
			(gnn.GINEConv(nn.Sequential(nn.Linear(33, hidden_dim_1), nn.LayerNorm(hidden_dim_1), nn.SiLU()), edge_dim=4), 'x, edge_index, edge_attr -> x'),
			(nn.LayerNorm(hidden_dim_1), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.conv2 = gnn.Sequential('x, edge_index, edge_attr', [
			(gnn.GINEConv(nn.Sequential(nn.Linear(hidden_dim_1, hidden_dim_2), nn.LayerNorm(hidden_dim_2), nn.SiLU()), edge_dim=4), 'x, edge_index, edge_attr -> x'),
			(nn.LayerNorm(hidden_dim_2), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim_2+33)*3, hidden_dim_3), nn.BatchNorm1d(hidden_dim_3), nn.SiLU(), nn.Dropout(0.3))
		self.layer_h_out = nn.Linear(hidden_dim_3, output_dim)


		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []


	def forward(self, batch):
		x, edge_index, edge_attr, _batch  = batch.x, batch.edge_index, batch.edge_attr, batch.batch

		types_emb = self.embed_types(x[:, 0].long())
		deg_emb = self.embed_deg(x[:, 1].long())
		hydrid_emb = self.embed_deg(x[:, 2].long())
		p_charge = x[:, 3].clone().to(torch.float32).reshape(-1, 1)
		charge_emb = self.embed_charge((x[:, 4]+10).long())
		arom = x[:, 5].clone().to(torch.float32).reshape(-1, 1)
		hcount_emb = self.embed_hcount((x[:, 6]).long())
		ring = x[:, 7].clone().to(torch.float32).reshape(-1, 1)
		ring_5 = x[:, 8].clone().to(torch.float32).reshape(-1, 1)
		ring_6 = x[:, 9].clone().to(torch.float32).reshape(-1, 1)
		chiral = self.embed_chiral((x[:, 10]).long())

		edge_attr = self.embed_edges(edge_attr.long())
		
		x = torch.cat((types_emb, deg_emb, hydrid_emb, p_charge, charge_emb, arom, hcount_emb, ring, ring_5, ring_6, chiral), dim=1)

		x0 = self.conv1(x, edge_index, edge_attr)
		x1 = self.conv2(x0, edge_index, edge_attr)

		x_all = torch.cat([x, x1], dim=-1)
		x_mean = gnn.global_mean_pool(x_all, _batch)
		x_max = gnn.global_max_pool(x_all, _batch)
		x_add = gnn.global_add_pool(x_all, _batch)

		x = torch.cat([x_mean, x_max, x_add], dim=-1)

		x = self.layer_gnn_h(x)
		out = self.layer_h_out(x)

		return out


class Attention(nn.Module):

	analysis = analysis
	fit = fit

	def __init__(self, hidden_dim_1, hidden_dim_2, output_dim):

		
		super().__init__()
		self.embed_types = nn.Embedding(120, 8)
		self.embed_hybrid = nn.Embedding(24, 4)
		self.embed_charge = nn.Embedding(12, 4)
		self.embed_hcount = nn.Embedding(12, 4)
		self.embed_deg = nn.Embedding(12, 4)
		self.embed_edges = nn.Embedding(6, 4)
		self.embed_chiral = nn.Embedding(12, 4)

		self.gnn = gnn.AttentiveFP(
            in_channels=33,
            hidden_channels=hidden_dim_1,
            out_channels=1,
            edge_dim=4,
            num_layers=3,
            num_timesteps=2,
            dropout=0.2
        )

		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim_1+33)*3, hidden_dim_2), nn.BatchNorm1d(hidden_dim_2), nn.SiLU(), nn.Dropout(0.2))
		self.layer_h_out = nn.Linear(hidden_dim_2, output_dim)


		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []


	def forward(self, batch):
		x, edge_index, edge_attr, _batch  = batch.x, batch.edge_index, batch.edge_attr, batch.batch

		types_emb = self.embed_types(x[:, 0].long())
		deg_emb = self.embed_deg(x[:, 1].long())
		hydrid_emb = self.embed_deg(x[:, 2].long())
		p_charge = x[:, 3].clone().to(torch.float32).reshape(-1, 1)
		charge_emb = self.embed_charge((x[:, 4]+10).long())
		arom = x[:, 5].clone().to(torch.float32).reshape(-1, 1)
		hcount_emb = self.embed_hcount((x[:, 6]).long())
		ring = x[:, 7].clone().to(torch.float32).reshape(-1, 1)
		ring_5 = x[:, 8].clone().to(torch.float32).reshape(-1, 1)
		ring_6 = x[:, 9].clone().to(torch.float32).reshape(-1, 1)
		chiral = self.embed_chiral((x[:, 10]).long())

		edge_attr = self.embed_edges(edge_attr.long())
		
		x = torch.cat((types_emb, deg_emb, hydrid_emb, p_charge, charge_emb, arom, hcount_emb, ring, ring_5, ring_6, chiral), dim=1)

		out = self.gnn(x, edge_index, edge_attr, _batch)

		return out

