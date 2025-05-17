import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Complex activation: modReLU
# -----------------------------------------------------------------------------
def modReLU(z: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mag   = torch.abs(z)
    scale = F.relu(mag + b) / (mag + eps)
    return z * scale

# -----------------------------------------------------------------------------
# Custom Complex‐MSE Loss (target is real, imag part zero)
# -----------------------------------------------------------------------------
def complex_mse_loss(pred: torch.Tensor, target: torch.Tensor, reduction='mean') -> torch.Tensor:
    loss_r = F.mse_loss(pred.real, target, reduction=reduction)
    loss_i = F.mse_loss(pred.imag, torch.zeros_like(pred.imag), reduction=reduction)
    return loss_r + loss_i

# -----------------------------------------------------------------------------
# ComplexLayerNorm, ComplexLinear, etc.
# -----------------------------------------------------------------------------
class ComplexLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.ln_r = nn.LayerNorm(dim, eps=eps)
        self.ln_i = nn.LayerNorm(dim, eps=eps)
    def forward(self, z):
        return torch.complex(self.ln_r(z.real), self.ln_i(z.imag))

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.r = nn.Linear(in_features, out_features, bias=bias)
        self.i = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        real = self.r(z.real) - self.i(z.imag)
        imag = self.r(z.imag) + self.i(z.real)
        return torch.complex(real, imag)

class ComplexMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.nh       = num_heads
        self.Wq = ComplexLinear(embed_dim, embed_dim)
        self.Wk = ComplexLinear(embed_dim, embed_dim)
        self.Wv = ComplexLinear(embed_dim, embed_dim)
        self.Wo = ComplexLinear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L, D = z.shape
        Q, K, V = self.Wq(z), self.Wk(z), self.Wv(z)
        def split(x):
            r = x.real.view(B, L, self.nh, self.head_dim).transpose(1,2)
            i = x.imag.view(B, L, self.nh, self.head_dim).transpose(1,2)
            return r, i
        Qr, Qi = split(Q)
        Kr, Ki = split(K)
        Vr, Vi = split(V)
        scores = (Qr@Kr.transpose(-2,-1) + Qi@Ki.transpose(-2,-1)) / (self.head_dim**0.5)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)
        Or = attn@Vr
        Oi = attn@Vi
        Or = Or.transpose(1,2).contiguous().view(B, L, D)
        Oi = Oi.transpose(1,2).contiguous().view(B, L, D)
        O  = torch.complex(Or, Oi)
        return self.Wo(O)

class ComplexTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.attn = ComplexMultiheadAttention(embed_dim,num_heads,dropout)
        self.norm1 = ComplexLayerNorm(embed_dim)
        self.norm2 = ComplexLayerNorm(embed_dim)
        self.ff1   = ComplexLinear(embed_dim, dim_feedforward)
        self.b1    = nn.Parameter(torch.zeros(dim_feedforward))
        self.ff2   = ComplexLinear(dim_feedforward, embed_dim)
        self.dropout_real = nn.Dropout(dropout)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.norm1(z)
        y = self.attn(y)
        z = z + y
        y = self.norm2(z)
        y = self.ff1(y)
        y = modReLU(y, self.b1.view(1,1,-1))
        y = self.ff2(y)
        yr = self.dropout_real(y.real)
        yi = self.dropout_real(y.imag)
        return z + torch.complex(yr, yi)

class ComplexTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            ComplexTransformerEncoderLayer(embed_dim,num_heads,dim_feedforward,dropout)
            for _ in range(num_layers)
        ])
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z)
        return z

class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.r_t = nn.ConvTranspose1d(in_ch,out_ch,kernel_size,stride,padding,output_padding,bias=bias)
        self.i_t = nn.ConvTranspose1d(in_ch,out_ch,kernel_size,stride,padding,output_padding,bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.r_t(x.real) - self.i_t(x.imag)
        imag = self.r_t(x.imag) + self.i_t(x.real)
        return torch.complex(real, imag)

class HybridEncoder(nn.Module):
    def __init__(self, input_dim=2, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.1, max_seq_len=2000):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim,64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(64,128,kernel_size=7,stride=2,padding=3),
            nn.ReLU(inplace=True),
        )
        self.proj = ComplexLinear(128,d_model)
        self.pos  = nn.Parameter(torch.randn(1,max_seq_len//4,d_model))
        self.trans= ComplexTransformerEncoder(d_model,nhead,num_layers,dim_feedforward,dropout)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0,2,1)
        x = self.cnn(x)
        x = x.permute(0,2,1)
        z = torch.complex(x, torch.zeros_like(x))
        z = self.proj(z)
        z = z + self.pos[:, :z.size(1)]
        return self.trans(z)

class HybridDecoder(nn.Module):
    def __init__(self, output_dim=2, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.1, seq_len=2000):
        super().__init__()
        self.trans_up  = ComplexTransformerEncoder(d_model,nhead,num_layers,dim_feedforward,dropout)
        self.proj_back = ComplexLinear(d_model,d_model)
        self.tconv1    = ComplexConvTranspose1d(d_model,d_model,7,2,3,1)
        self.b1        = nn.Parameter(torch.zeros(d_model))
        self.tconv2    = ComplexConvTranspose1d(d_model,output_dim,7,2,3,1)
        self.b2        = nn.Parameter(torch.zeros(output_dim))
    def forward(self,z: torch.Tensor) -> torch.Tensor:
        z = self.trans_up(z)
        z = self.proj_back(z)
        x = z.permute(0,2,1)
        x = self.tconv1(x)
        x = modReLU(x, self.b1.view(1,-1,1))
        x = self.tconv2(x)
        x = modReLU(x, self.b2.view(1,-1,1))
        return x.permute(0,2,1)

class HybridAutoencoder(nn.Module):
    def __init__(self, seq_len=2000, input_dim=2, d_model=256, nhead=8,
                 num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.enc = HybridEncoder(input_dim,d_model,nhead,num_layers,dim_feedforward,dropout,seq_len)
        self.dec = HybridDecoder(input_dim,d_model,nhead,num_layers,dim_feedforward,dropout,seq_len)
    def forward(self,x):
        return self.dec(self.enc(x))

def train(model, loader, device, epochs=20, lr=1e-4):
    model.to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    scaler = torch.amp.GradScaler()

    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for batch in loader:
            x, = batch
            x = x.to(device)
            optim.zero_grad()
            # fixed: pass device_type
            with torch.amp.autocast(device_type=device.type):
                x_hat = model(x)
                loss  = complex_mse_loss(x_hat, x, reduction='sum')
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            total += loss.item()
        sched.step()
        print(f"Epoch {ep:02d} — Avg MSE: {total/len(loader):.6f}")

def compute_anomaly_scores(model, loader, device):
    model.to(device).eval()
    errs = []
    with torch.no_grad():
        for batch in loader:
            x, = batch
            x = x.to(device)
            x_hat = model(x)
            mse_r = (x_hat.real - x)**2
            mse_i = (x_hat.imag)**2
            mse   = mse_r + mse_i
            seq_err = mse.reshape(mse.size(0), -1).mean(dim=1)
            errs.append(seq_err.cpu())
    return torch.cat(errs, dim=0)

def build_loader(df: pd.DataFrame, seq_len: int, batch_size: int, shuffle: bool):
    signals = []
    for arr in df['signal'].values:
        x = np.array(arr, dtype=np.complex64)
        if x.shape[0] >= seq_len:
            x = x[:seq_len]
        else:
            pad = np.zeros(seq_len - x.shape[0], dtype=np.complex64)
            x = np.concatenate([x, pad])
        stacked = np.stack([x.real, x.imag], axis=1)
        signals.append(stacked.astype(np.float32))
    tensor = torch.tensor(np.stack(signals, axis=0))
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    onBody     = pd.read_pickle('dataset/onBody.pkl')
    onBody_Val = pd.read_pickle('dataset/onBody_Val.pkl')
    dfAnomoly  = pd.read_pickle('dataset/dfAnomoly.pkl')
    if 'freq_dev' in onBody.columns:
        onBody.rename(columns={'freq_dev':'signal'}, inplace=True)
    if 'freq_dev' in onBody_Val.columns:
        onBody_Val.rename(columns={'freq_dev':'signal'}, inplace=True)
    if isinstance(dfAnomoly, pd.Series):
        dfAnomoly = dfAnomoly.to_frame(name='signal')

    batch_size, seq_len, input_dim = 32, 2000, 2
    train_loader = build_loader(onBody, seq_len, batch_size, True)
    val_loader   = build_loader(onBody_Val, seq_len, batch_size, False)
    anom_loader  = build_loader(dfAnomoly, seq_len, batch_size, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HybridAutoencoder(seq_len, input_dim).to(device)

    train(model, train_loader, device, epochs=20, lr=1e-4)

    val_scores  = compute_anomaly_scores(model, val_loader,   device)
    anom_scores = compute_anomaly_scores(model, anom_loader,  device)

    print("Errors on VALIDATION (normal):", val_scores.numpy())
    print("Errors on ANOMALY set:        ", anom_scores.numpy())

    labels = torch.cat([torch.zeros_like(val_scores), torch.ones_like(anom_scores)])
    scores = torch.cat([val_scores, anom_scores])

    best_acc, best_t, best_c = 0.0, 0.0, (0,0,0,0)
    for t in np.arange(0.003, 0.00801, 1e-5):
        preds = (scores > t).int()
        acc   = (preds == labels.int()).float().mean().item()
        if acc > best_acc:
            tp = ((preds==1)&(labels==1)).sum().item()
            tn = ((preds==0)&(labels==0)).sum().item()
            fp = ((preds==1)&(labels==0)).sum().item()
            fn = ((preds==0)&(labels==1)).sum().item()
            best_acc, best_t, best_c = acc, t, (tp, tn, fp, fn)

    tp, tn, fp, fn = best_c
    print(f"\nBest threshold: {best_t:.5f}")
    print(f"Best accuracy:   {best_acc*100:.2f}%  → TP={tp}, TN={tn}, FP={fp}, FN={fn}")

