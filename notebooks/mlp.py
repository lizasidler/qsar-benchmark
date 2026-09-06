import torch 
from torch import nn
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


def analysis(self, train, act_train, val, act_val, device):

	pred = self(torch.tensor(train, dtype=torch.float32).to(device))
	pred = pred.cpu().detach().numpy()

	r2 = r2_score(act_train, pred)
	rmse = np.sqrt(mean_squared_error(act_train, pred))
	mae = mean_absolute_error(act_train, pred)

	print(f"R² Train Score: {r2:.3f}")
	print(f"Train RMSE:     ±{rmse:.3f}")
	print(f"Train MAE:      ±{mae:.3f}")

	pred = self(torch.tensor(val, dtype=torch.float32).to(device))
	pred = pred.cpu().detach().numpy()

	r2 = r2_score(act_val, pred)
	rmse = np.sqrt(mean_squared_error(act_val, pred))
	mae = mean_absolute_error(act_val, pred)

	print(f"R² Val Score: {r2:.3f}")
	print(f"Val RMSE:     ±{rmse:.3f}")
	print(f"Val MAE:      ±{mae:.3f}")


class MLP_1(nn.Module):

	analysis = analysis

	def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):

		super().__init__()
		self.layer_in_h1 = nn.Sequential(nn.Linear(input_dim, hidden_dim_1), nn.BatchNorm1d(hidden_dim_1), nn.SiLU())
		self.layer_h1_h2 = nn.Sequential(nn.Linear(hidden_dim_1, hidden_dim_2), nn.BatchNorm1d(hidden_dim_2), nn.SiLU())
		self.layer_h2_h3 = nn.Sequential(nn.Linear(hidden_dim_2, hidden_dim_3), nn.BatchNorm1d(hidden_dim_3), nn.SiLU())
		self.layer_h3_out = nn.Linear(hidden_dim_3, output_dim)

		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

		self.train_loss = []
		self.val_loss = []


	def forward(self, x):

		x = self.layer_in_h1(x)
		x = self.layer_h1_h2(x)
		x = self.layer_h2_h3(x)
		out = self.layer_h3_out(x)

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


class MLP_2(nn.Module):

	analysis = analysis

	def __init__(self, MLP_1_dim, MLP_2_dim, comb_dim):

		super().__init__()


		self.layer_in_h1_1 = nn.Sequential(nn.Linear(MLP_1_dim[0], MLP_1_dim[1]), nn.BatchNorm1d(MLP_1_dim[1]), nn.SiLU())
		self.layer_h1_h2_1 = nn.Sequential(nn.Linear(MLP_1_dim[1], MLP_1_dim[2]), nn.BatchNorm1d(MLP_1_dim[2]), nn.SiLU())
		self.layer_h2_h3_1 = nn.Sequential(nn.Linear(MLP_1_dim[2], MLP_1_dim[3]), nn.BatchNorm1d(MLP_1_dim[3]), nn.SiLU(), nn.Dropout(p=0.2))

		self.layer_in_h1_2 = nn.Sequential(nn.Linear(MLP_2_dim[0], MLP_2_dim[1]), nn.BatchNorm1d(MLP_2_dim[1]), nn.SiLU())
		self.layer_h1_h2_2 = nn.Sequential(nn.Linear(MLP_2_dim[1], MLP_2_dim[2]), nn.BatchNorm1d(MLP_2_dim[2]), nn.SiLU())
		self.layer_h2_h3_2 = nn.Sequential(nn.Linear(MLP_2_dim[2], MLP_2_dim[3]), nn.BatchNorm1d(MLP_2_dim[3]), nn.SiLU(), nn.Dropout(p=0.2))


		self.layer_h_h_12 = nn.Sequential(nn.Linear(MLP_1_dim[3]+MLP_2_dim[3], comb_dim[0]), nn.BatchNorm1d(comb_dim[0]), nn.SiLU(), nn.Dropout(p=0.0))
		self.layer_h_12_out = nn.Linear(comb_dim[0], comb_dim[1])

		self.loss_fn = nn.SmoothL1Loss(beta=0.1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

		self.train_loss = []
		self.val_loss = []


	def forward(self, x1, x2):

		x1 = self.layer_in_h1_1(x1)
		x1 = self.layer_h1_h2_1(x1)
		x1 = self.layer_h2_h3_1(x1)

		x2 = self.layer_in_h1_2(x2)
		x2 = self.layer_h1_h2_2(x2)
		x2 = self.layer_h2_h3_2(x2)

		x = torch.cat((x1, x2), dim=1)
		x = self.layer_h_h_12(x)

		out = self.layer_h_12_out(x)

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




