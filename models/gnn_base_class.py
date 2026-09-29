import torch
import torch.nn as nn 
import torch_geometric.nn as gnn 
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


class BaseClass(nn.Module):

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


	def fit(self, train_loader, val_loader, device, epochs=100, tol=0.4):

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

			if t>50 and np.std(self.val_loss[-50:-1])<tol:
				break


	def forward(self, batch):
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