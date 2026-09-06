import torch
import torch.nn as nn 
import torch_geometric.nn as gnn 
from tqdm import tqdm


class GatedEdge(gnn.MessagePassing):
	def __init__(self, hidden_dim, num_steps=3):
		super().__init__(aggr='add')  # "Add" aggregation (Step 5).

		self.edge_weight = nn.Sequential(nn.Linear(6, 1), nn.Tanh())
		self.gru = nn.GRUCell(hidden_dim, hidden_dim)
		self.num_steps = num_steps

	def forward(self, x, edge_index, edge_attr):

		# Step 4-5: Start propagating messages.

		for _ in range(self.num_steps):

			out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
			x = self.gru(out, x)

		return x

	def message(self, x_j, edge_attr):
	    
	    edge_attr = self.edge_weight(edge_attr)

	    return  x_j * (1.0 + edge_attr)

class GatedEdgeConv(nn.Module):

	def __init__(self, hidden_dim, output_dim):

		super().__init__()
		self.embed_types = nn.Embedding(120, 8)
		self.embed_hybrid = nn.Embedding(24, 4)
		self.embed_charge = nn.Embedding(12, 4)
		self.embed_hcount = nn.Embedding(12, 4)
		self.embed_deg = nn.Embedding(12, 4)
		self.embed_edges = nn.Embedding(6, 6)

		self.p_charge_nn = nn.Linear(1, 4)

		self.node_init = nn.Sequential(nn.Linear(26, hidden_dim), nn.SiLU())

		#GCNonv nope

		self.conv1 = gnn.Sequential('x, edge_index, edge_attr', [
			(GatedEdge(hidden_dim), 'x, edge_index, edge_attr -> x'),
			(nn.LayerNorm(hidden_dim), 'x -> x'),
			(nn.SiLU(), 'x -> x'),
			])

		self.layer_gnn_h = nn.Sequential(nn.Linear((hidden_dim+26)*3, 64), nn.BatchNorm1d(64), nn.SiLU())
		self.layer_h_out = nn.Linear(64, output_dim)


		self.loss_fn = nn.HuberLoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []


	def forward(self, batch):
		x, edge_index, edge_attr, _batch  = batch.x, batch.edge_index, batch.edge_attr, batch.batch

		types_emb = self.embed_types(x[:, 0].long())
		deg_emb = self.embed_deg(x[:, 1].long())
		hydrid_emb = self.embed_deg(x[:, 2].long())
		#p_charge = self.p_charge_nn(x[:, 3:4])
		p_charge = torch.tensor(x[:, 3].reshape(-1, 1), dtype=torch.float32)
		charge_emb = self.embed_charge((x[:, 4]+10).long())
		arom = torch.tensor(x[:, 5].reshape(-1, 1), dtype=torch.float32)
		hcount_emb = self.embed_hcount((x[:, 6]).long())
		edge_attr = self.embed_edges(edge_attr.long())


		x = torch.cat((types_emb, deg_emb, hydrid_emb, p_charge, charge_emb, arom, hcount_emb), dim=1)

		x00 = self.node_init(x)

		x0 = self.conv1(x00, edge_index, edge_attr)
		#x1 = self.conv1(x0, edge_index, edge_attr)

		x_all = torch.cat([x, x0], dim=-1)
		x_mean = gnn.global_mean_pool(x_all, _batch)
		x_max = gnn.global_max_pool(x_all, _batch)
		x_add = gnn.global_add_pool(x_all, _batch)

		x = torch.cat([x_mean, x_max, x_add], dim=-1)

		#print("Pooled vector variance across batch:", x.std(dim=0).mean().item())
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
				#train_y = (train_y - 4) / 4
				print(pred[0], train_y[0])
				r = torch.corrcoef(torch.stack([pred.view(-1), train_y.view(-1)]))[0, 1]
				#print(f"Pearson r: {r.item():.4f}")
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
			    	#loss = self.loss_fn(pred, (batch.y-4)/4)
			    	loss = self.loss_fn(pred, batch.y)

			    	running_val_loss += loss.item() * batch.x.size(0)

			running_val_loss /= len(val_loader.dataset)
			self.val_loss.append(running_val_loss)


