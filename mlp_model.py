import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import init_weights

SMALL = 1e-16


class VAE_Concrete_Simulated(nn.Module):
    def __init__(self, bottleneck_size=100, temp=1., alpha0=100., dataset='Simulated',
                 hidden=500, X_dim = 2952, y_dim = 10, learning_type='supervised'):
        super(VAE_Concrete_Simulated, self).__init__()
        self.temp = temp
        self.dataset = dataset
        self.bottleneck_size = bottleneck_size
        self.learning_type = learning_type

        self.fc1_encode = nn.Linear(X_dim, hidden)
        self.batchNorm1 = nn.BatchNorm1d(hidden)

        self.fc2_encode = nn.Linear(hidden, self.bottleneck_size * 2)

        self.fc2_encode_mean = nn.Linear(self.bottleneck_size*2, self.bottleneck_size)
        self.fc2_encode_logvar = nn.Linear(self.bottleneck_size*2, self.bottleneck_size)

        self.fc2_encode_y = nn.Linear(hidden, y_dim)

        if self.learning_type == 'supervised':
            self.fc1_decode = nn.Linear(self.bottleneck_size + y_dim, hidden)
        elif self.learning_type == 'unsupervised':
            self.fc1_decode = nn.Linear(self.bottleneck_size, hidden)

        self.fc2_decode = nn.Linear(hidden, X_dim)

    # 2952 -> 500 -> 200
    def encode(self, x):
        encode_1 = F.relu(self.batchNorm1(self.fc1_encode(x)))
        encode_2 = self.fc2_encode(encode_1)
        mu = self.fc2_encode_mean(encode_2)
        logvar = self.fc2_encode_logvar(encode_2)

        # mu, logvar = torch.split(encode_2, self.bottleneck_size, 1)

        y_logit = self.fc2_encode_y(encode_1)
        y_logit = F.softmax(y_logit, dim=1)  # 200 -> 10

        return mu, logvar, y_logit, encode_2
        # return mu, logvar, y_logit


    def decode(self, z_discrete, learning_type):
        decode_1 = F.relu(self.fc1_decode(z_discrete))

        decode_2 = self.fc2_decode(decode_1)
        return decode_2

    def reparametrize_gaussian(self, mu, logvar):
        # noise = torch.autograd.Variable(mu.data.clone().normal_(0, 1), requires_grad=False)
        # return mu + (noise * logvar.exp())

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, learning_type):
        mu, logvar, y_logit, _ = self.encode(x)

        z_continuous = self.reparametrize_gaussian(mu, logvar)

        if learning_type == 'supervised':
            inputForDecoder = torch.cat([z_continuous, y_logit], 1)
        elif learning_type == 'unsupervised':
            inputForDecoder = z_continuous

        recon_batch = self.decode(inputForDecoder, learning_type)

        return mu, logvar, y_logit, recon_batch
