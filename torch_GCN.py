import torch
import torch.nn as nn 
import torch_geometric.nn as gnn 
from tqdm import tqdm


class GCN(nn.Module):

	def __init__(self, input_dim, hidden_dim, output_dim):

		super().__init__()
		self.embed_at_types = nn.Embedding(120, input_dim)
		self.layer_in_gnn = gnn.Sequential('x, edge_index, batch', [
			(gnn.GCNConv(input_dim, hidden_dim), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			(nn.Dropout(p=0.1), 'x -> x'),
			(gnn.GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
			(nn.LayerNorm(hidden_dim), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			(nn.Dropout(p=0.1), 'x -> x'),
			(gnn.global_mean_pool, 'x, batch -> x')
			])
		self.layer_gnn_h = nn.Sequential(nn.Linear(hidden_dim, 32), nn.BatchNorm1d(32), nn.SiLU())
		self.layer_h_out = nn.Linear(32, output_dim)


		self.loss_fn = nn.HuberLoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []


	def forward(self, batch):
		x, edge_index, _batch  = batch.x, batch.edge_index, batch.batch

		x = self.embed_at_types(x.long())
		x = self.layer_in_gnn(x, edge_index, _batch)
		x = self.layer_gnn_h(x)
		out = self.layer_h_out(x)

		return out


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
				# 2. Normalize targets (Mean = 0, Std = 1)
				y_norm = (train_y - 4) / 4
				loss = self.loss_fn(pred, y_norm)
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
			    	loss = self.loss_fn(pred, (batch.y-4)/4)

			    	running_val_loss += loss.item() * batch.x.size(0)

			running_val_loss /= len(val_loader.dataset)
			self.val_loss.append(running_val_loss)
		





