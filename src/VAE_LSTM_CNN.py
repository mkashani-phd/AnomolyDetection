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
    



class CNNLSTMEmbeddingNet(nn.Module):
    def __init__(self, input_length, num_channels, embedding_dim=64):
        super(CNNLSTMEmbeddingNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Calculate the output size after the convolutional layers
        conv_output_size = input_length // 2 // 2  # Two pooling layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        
        # Fully connected layer to produce embeddings
        self.fc = nn.Linear(64 * conv_output_size, embedding_dim)
        
    def forward(self, x):
        # Ensure x has three dimensions: (batch_size, sequence_length, num_channels)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Adds a third dimension at the end (num_channels=1)

        # Correcting the initial transpose operation to fit Conv1d expectation
        x = x.transpose(1, 2)  # Now shape (batch_size, num_channels=1, sequence_length)
        
        # Convolutional layers with ReLU and pooling
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

class EnhancedCNNBiLSTM(nn.Module):
    def __init__(self, input_length, num_channels, embedding_dim=128):
        super(EnhancedCNNBiLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=100, padding=50)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=100, padding=50)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.1)

        # Adding LSTM layer. The number of input features to the LSTM is the number of output channels from the last conv layer.
        # Assuming the output from the pooling layer is properly reshaped for the LSTM input.
        conv_output_size = input_length // 4  # Adjust based on your pooling and convolution strides
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)

        # The output of LSTM is (batch_size, seq_len, num_directions * hidden_size)
        # Flatten LSTM output and feed into the fully connected layer to get embeddings
        self.fc = nn.Linear(conv_output_size * 128, embedding_dim)  # Adjust the input feature size accordingly

    def forward(self, x):
        # Ensure x has three dimensions: (batch_size, sequence_length, num_channels)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Adds a third dimension at the end (num_channels=1)
        x = x.transpose(1, 2)  # Assuming input shape is (batch_size, num_channels, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # Adjust for LSTM, shape becomes (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)

        # Flatten the output for the fully connected layer
        lstm_out_flattened = lstm_out.reshape(lstm_out.shape[0], -1)

        # Fully connected layer to produce embeddings
        embeddings = self.fc(lstm_out_flattened)

        return embeddings
