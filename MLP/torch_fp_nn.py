import torch 
from torch import nn
from tqdm import tqdm


class MLP(nn.Module):

	def __init__(self, input_dim_1, input_dim_2, output_dim):

		super().__init__()
		self.layer_in_h_1 = nn.Sequential(nn.Linear(input_dim_1, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(p=0.02))
		self.layer_h_h_1 = nn.Sequential(nn.Linear(256, 124), nn.BatchNorm1d(124), nn.SiLU(), nn.Dropout(p=0.2))
		self.layer_h_h_12 = nn.Sequential(nn.Linear(124, 32), nn.BatchNorm1d(32), nn.SiLU(), nn.Dropout(p=0.2))

		self.layer_in_h_2 = nn.Sequential(nn.Linear(input_dim_2, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(p=0.0))
		self.layer_h_h_2 = nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(), nn.Dropout(p=0.0))


		self.layer_h_h = nn.Sequential(nn.Linear(64, 16), nn.BatchNorm1d(16), nn.SiLU(), nn.Dropout(p=0.0))
		self.layer_h_out = nn.Linear(16, output_dim)

		self.loss_fn = nn.HuberLoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

		self.train_loss = []
		self.val_loss = []


	def forward(self, x1, x2):

		x1 = self.layer_in_h_1(x1)
		x1 = self.layer_h_h_1(x1)
		x1 = self.layer_h_h_12(x1)

		x2 = self.layer_in_h_2(x2)
		x2 = self.layer_h_h_2(x2)

		x = torch.cat((x1, x2), dim=1)
		x = self.layer_h_h(x)

		out = self.layer_h_out(x)

		return out


	def fit(self, train_loader, val_loader, device, epochs=100):

		self.train()
		running_train_loss = 0.0
		running_val_loss = 0.0

		for t in tqdm(range(epochs)):

			for (x1_train, x2_train, y_train) in train_loader:
				x1, x2, y = x1_train.to(device), x2_train.to(device), y_train.to(device)

				self.optimizer.zero_grad()

				pred = self(x1, x2)
				loss = self.loss_fn(pred, y)
				loss.backward()
				self.optimizer.step()

				running_train_loss += loss.item() * x1.size(0)

			running_train_loss /= len(train_loader.dataset)
			self.train_loss.append(running_train_loss)


			self.eval()

			with torch.no_grad():
			    for (x1_val, x2_val, y_val) in val_loader:
			    	x1, x2, y = x1_val.to(device), x2_val.to(device), y_val.to(device)

			    	pred = self(x1, x2)
			    	loss = self.loss_fn(pred, y)

			    	running_val_loss += loss.item() * x1.size(0)

			running_val_loss /= len(val_loader.dataset)
			self.val_loss.append(running_val_loss)
		





