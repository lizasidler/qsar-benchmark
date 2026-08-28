import torch 
from torch import nn
from tqdm import tqdm


class MLP(nn.Module):

	def __init__(self, input_dim, hidden_dim, output_dim):

		super().__init__()
		self.layer_in_h = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU(), nn.Dropout(p=0.0))
		self.layer_h_h = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU(), nn.Dropout(p=0.0))
		self.layer_h_out = nn.Linear(hidden_dim, output_dim)

		self.loss_fn = nn.HuberLoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

		self.train_loss = []
		self.val_loss = []


	def forward(self, x):

		x = self.layer_in_h(x)
		x = self.layer_h_h(x)
		out = self.layer_h_out(x)

		return out


	def fit(self, train_loader, val_loader, device, epochs=100):

		self.train()
		running_train_loss = 0.0
		running_val_loss = 0.0

		for t in tqdm(range(epochs)):

			for (x_train, y_train) in train_loader:
				x, y = x_train.to(device), y_train.to(device)

				self.optimizer.zero_grad()

				pred = self(x)
				loss = self.loss_fn(pred, y)
				loss.backward()
				self.optimizer.step()

				running_train_loss += loss.item() * x.size(0)

			running_train_loss /= len(train_loader.dataset)
			self.train_loss.append(running_train_loss)


			self.eval()

			with torch.no_grad():
			    for (x_val, y_val) in val_loader:
			    	x, y = x_val.to(device), y_val.to(device)

			    	pred = self(x)
			    	loss = self.loss_fn(pred, y)

			    	running_val_loss += loss.item() * x.size(0)

			running_val_loss /= len(val_loader.dataset)
			self.val_loss.append(running_val_loss)
		





