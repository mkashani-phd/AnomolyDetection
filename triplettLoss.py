import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# -----------------------------------------------------------------------------
# modReLU activation for complex inputs
# -----------------------------------------------------------------------------
def modReLU(z: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # z: complex64 tensor; b: real bias tensor (broadcastable)
    mag = torch.abs(z)
    scale = F.relu(mag + b) / (mag + eps)
    return z * scale

# -----------------------------------------------------------------------------
# Complex Linear layer: y = W·x + b, all complex
# -----------------------------------------------------------------------------
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # real and imag parts as separate real-valued Linear layers
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: complex tensor of shape [batch, in_features]
        xr, xi = x.real, x.imag
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        return torch.complex(yr, yi)

# -----------------------------------------------------------------------------
# Define the Complex Autoencoder
# -----------------------------------------------------------------------------
class ComplexAutoencoder(nn.Module):
    def __init__(self, dim=2000, hidden_dims=(512,128)):
        super().__init__()
        # encoder
        self.enc1 = ComplexLinear(dim, hidden_dims[0])
        self.b1   = nn.Parameter(torch.zeros(hidden_dims[0]))  # bias for modReLU
        self.enc2 = ComplexLinear(hidden_dims[0], hidden_dims[1])
        self.b2   = nn.Parameter(torch.zeros(hidden_dims[1]))
        # decoder
        self.dec1 = ComplexLinear(hidden_dims[1], hidden_dims[0])
        self.b3   = nn.Parameter(torch.zeros(hidden_dims[0]))
        self.dec2 = ComplexLinear(hidden_dims[0], dim)
        self.b4   = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encode
        z = modReLU(self.enc1(x), self.b1)
        z = modReLU(self.enc2(z), self.b2)
        # decode
        z = modReLU(self.dec1(z), self.b3)
        x_rec = self.dec2(z)  # no activation on final layer
        return x_rec

# -----------------------------------------------------------------------------
# Load your datasets
# -----------------------------------------------------------------------------
onBody     = pd.read_pickle('dataset/onBody.pkl')
onBody_Val = pd.read_pickle('dataset/onBody_Val.pkl')
dfAnomoly  = pd.read_pickle('dataset/dfAnomoly.pkl')

for df in (onBody, onBody_Val):
    if 'freq_dev' in df.columns:
        df.rename(columns={'freq_dev':'signal'}, inplace=True)

# ensure anomaly as DataFrame
if isinstance(dfAnomoly, pd.Series):
    dfAnomoly = dfAnomoly.to_frame(name='signal')

# stack into tensors, assume each row is one 2000-length complex sequence
# stack into tensors, cropping each sequence to exactly length=2000
def df_to_tensor(df, seq_len=2000):
    # df['signal'] should be an array‐like of complex values per row
    arr = torch.stack([
        torch.tensor(sig[:seq_len], dtype=torch.cfloat)
        for sig in df['signal'].values
    ])
    return arr

train_tensor = df_to_tensor(onBody,     seq_len=2000)
val_tensor   = df_to_tensor(onBody_Val, seq_len=2000)
anom_tensor  = df_to_tensor(dfAnomoly,  seq_len=2000)


train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_tensor),   batch_size=64)
anom_loader  = DataLoader(TensorDataset(anom_tensor),  batch_size=64)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = ComplexAutoencoder(dim=2000).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
mse    = nn.MSELoss()

for epoch in range(1, 51):
    model.train()
    tot = 0.0
    for (x_batch,) in train_loader:
        x_batch = x_batch.to(device)
        x_rec   = model(x_batch)
        loss    = mse(x_rec.real, x_batch.real) + mse(x_rec.imag, x_batch.imag)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * x_batch.size(0)
    print(f"Epoch {epoch:02d}, Train Loss = {tot/len(train_tensor):.4e}")

# -----------------------------------------------------------------------------
# Compute reconstruction errors on validation to set threshold
# -----------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    rec_errors = []
    for (x_val,) in val_loader:
        x_val  = x_val.to(device)
        x_rec  = model(x_val)
        err    = torch.sum((x_rec.real - x_val.real)**2 + (x_rec.imag - x_val.imag)**2, dim=1)
        rec_errors.append(err.cpu())
    rec_errors = torch.cat(rec_errors)
thresh = rec_errors.mean() #+ 3 * rec_errors.std()
print("Anomaly threshold set at:", thresh.item())

# -----------------------------------------------------------------------------
# Evaluate on anomalies
# -----------------------------------------------------------------------------
def compute_rate(dataloader, model, threshold, device='cpu'):
    model.eval()
    errs = []
    with torch.no_grad():
        for (x_batch,) in dataloader:
            x_batch = x_batch.to(device)
            x_rec   = model(x_batch)
            # per‐sequence MSE error
            err = torch.sum((x_rec.real - x_batch.real)**2 +
                            (x_rec.imag - x_batch.imag)**2, dim=1)
            errs.append(err.cpu())
    errs = torch.cat(errs)
    # fraction above threshold
    return float((errs > threshold).float().mean())

# after you've set `thresh`:



det_rate = compute_rate(anom_loader,  model, thresh, device)
fpr      = compute_rate(val_loader,   model, thresh, device)

print(f"Anomaly detection rate: {det_rate*100:.2f}%")
print(f"False‐positive rate:  {fpr*100:.2f}%")


import matplotlib.pyplot as plt

thresholds = np.arange(0.0001, 5.0, 0.001)
det_rates = []
fprs = []

for t in thresholds:
    det_rates.append(compute_rate(anom_loader, model, t, device))
    fprs.append(compute_rate(val_loader, model, t, device))

plt.figure(figsize=(8,5))
plt.plot(thresholds, det_rates, label='Detection Rate (Anomaly)')
plt.plot(thresholds, fprs, label='False Positive Rate (Validation)')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.title('Detection Rate vs Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()