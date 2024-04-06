import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvVAE(nn.Module):
    def __init__(self, seq_length, latent_dim):
        super(ConvVAE, self).__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc1 = nn.Linear(self._conv_output(seq_length), 256)
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, self._conv_output(seq_length)),
            nn.ReLU(),
        )
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1)

    def _conv_output(self, size):
        size = size // 2 // 2  # Two conv layers with stride 2
        return size * 64  # Adjust based on the number of output channels of the last conv layer

    def encode(self, x):
        conv_out = self.encoder(x)
        h = F.relu(self.fc1(conv_out))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = F.relu(self.decoder_input(z))
        deconv_input = self.decoder(h).view(-1, 64, self.seq_length // 4)
        deconv_out = F.relu(self.deconv1(deconv_input))
        return torch.sigmoid(self.deconv2(deconv_out))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    




def vae_loss(recon_x, x, mu, logvar, seq_length):
    BCE = nn.functional.mse_loss(recon_x, x.view(-1,1, seq_length), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, seq_length, latent_dim):
        super(Autoencoder, self).__init__()
        self.input_length = seq_length
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(seq_length, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, seq_length),
            nn.Tanh()  # Using Tanh to ensure output is between -1 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

