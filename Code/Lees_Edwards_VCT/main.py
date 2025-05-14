import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from files.data_preprocess import Data_Preprocess
from files.model import Model

# Reading the data
dp = Data_Preprocess(path="raw_data_Lees-Edwards_BC.tgz")

# Getting the inputs and outputs
X, y, time_X, time_y, box_length = dp.get_data(output_timesteps=10)
G_fixed = dp.get_G_fixed(X, box_length, time_X, d_threshold=3*2.5e-6,
                            shear_rate=1000, alpha=0.001)
X_transformed, last_location, y_transformed = dp.transform_data(X, y, time_X, time_y, box_length, shear_rate=1000)

batch_size = 1

print(f"X_transformed : {X_transformed.shape}, G_fixed : {G_fixed.shape}") 
print(f"last_location : {last_location.shape}, y_transformed : {y_transformed.shape}")

# Check available GPUs
print("Available GPUs:", torch.cuda.device_count())
# Set desired GPU
torch.cuda.set_device(0)  # Assigning second GPU

# Settings for intializing the model
particles = X_transformed.shape[-1]
conv_in_channels = X_transformed.shape[1]
conv_out_channels = 64
conv_kernel_size = (3, 1)
seq_how_many = 2
seq_hidden_size = 350
seq_decoder_input_size = last_location.shape[-1]
seq_num_layers = 2
seq_output_size = X_transformed.shape[1]
seq_timesteps = y_transformed.shape[2]
conv_stride = 1
conv_dropout=0.0
conv_residual=True
seq_encoder_dropout=0.0
seq_decoder_dropout=0.0
seq_decoder_isDropout=False

# Epochs to train for
epochs = 6000

model = Model(particles, conv_in_channels, conv_out_channels, conv_kernel_size,
              seq_how_many, seq_hidden_size, seq_decoder_input_size, seq_num_layers, seq_output_size, seq_timesteps,
              conv_stride = conv_stride, conv_dropout=conv_dropout, conv_residual=conv_residual,
              seq_encoder_dropout=seq_encoder_dropout, 
              seq_decoder_dropout=seq_decoder_dropout, seq_decoder_isDropout=seq_decoder_isDropout)
              
class Dataset_class(torch.utils.data.Dataset):
    def __init__(self, x, g, l, y):
        self.x = torch.from_numpy(x).float()
        self.g = torch.from_numpy(g).float()
        self.l = torch.from_numpy(l).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.g[i], self.l[i], self.y[i]

# initialize the dataset, dataloader and loss function
dataset = Dataset_class(X_transformed, G_fixed, last_location, y_transformed)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
loss_fn = nn.MSELoss()
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"

def get_train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0.0
    for data in dataloader:
        x_train, g_fixed, last_location, y_train = data
        x_train = x_train.to(DEVICE)
        g_fixed = g_fixed.to(DEVICE)
        last_location = last_location.to(DEVICE)
        y_train = y_train.to(DEVICE)
        
        pred = model(x_train, g_fixed, last_location)
        loss = loss_fn(pred, y_train)
    
        #optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= num_batches
    return train_loss

# Training loop
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 0.0)         
training_loss = []
least_cost = np.array([100.5])
for j in range(epochs):
    train_loss = get_train(dataloader, model, loss_fn, optimizer)
    training_loss.append(train_loss)
    if train_loss < least_cost:
        torch.save(model.state_dict(), "best_model.pt")
    if (j+1)%10 == 0:
        print(f"Iteration : {j+1}, Train loss : {train_loss:0.6f}")

train_loss = np.array(training_loss)

# Predicting the results
best_model = Model(particles, conv_in_channels, conv_out_channels, conv_kernel_size,
              seq_how_many, seq_hidden_size, seq_decoder_input_size, seq_num_layers, seq_output_size, seq_timesteps,
              conv_stride = conv_stride, conv_dropout=conv_dropout, conv_residual=conv_residual,
              seq_encoder_dropout=seq_encoder_dropout, 
              seq_decoder_dropout=seq_decoder_dropout, seq_decoder_isDropout=seq_decoder_isDropout)
              
best_model.load_state_dict(torch.load("best_model.pt"))
best_model.eval()
DEVICE = torch.device('cpu')
dX_torch = torch.from_numpy(X_transformed).float().to(DEVICE)
G_fixed_torch = torch.from_numpy(G_fixed).float().to(DEVICE)
last_location_torch = torch.from_numpy(last_location).float().to(DEVICE)
best_model = best_model.to(DEVICE)
dy_pred = best_model.forward(dX_torch, G_fixed_torch, last_location_torch)

y_pred = dp.get_added_pred(last_location_torch, dy_pred)
y_pred = dp.inverse_transform(y_pred, box_length).detach().numpy()
dy_pred = dy_pred.detach().numpy()

rmse = dp.compute_rmse(y, y_pred, box_length, time_y, shear_rate=1000)

# plotting the parity plot
y_temp_true = y.reshape(-1,)
y_temp_pred = y_pred.reshape(-1,)

with open('predictions.npy', 'wb') as f:
    np.save(f, dy_pred)
    np.save(f, y_pred)
    np.save(f, train_loss)
    np.save(f, rmse)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(y_temp_true, y_temp_pred)
ax[0].set_xlabel("True")
ax[0].set_ylabel("Prediction")
ax[1].plot(train_loss)
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Training loss")
ax[1].set_yscale("log")
plt.savefig("Parity_plot.png")
plt.tight_layout()
plt.close()