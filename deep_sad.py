import torch 
from torch import nn, optim
import torch.nn.functional as F
import logging 
import time
import numpy as np
from sklearn.metrics import roc_auc_score

class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None

    def forward(self, *input):
        raise NotImplementedError


class Linear_BN_leakyReLU(nn.Module):
    """Linear layer followed by BatchNorm1d and leaky ReLU."""

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))
        # return self.leaky_relu(self.linear(x))


class MLP(BaseNet):
    """MLP Encoder for Deep-SAD."""

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()
        self.rep_dim = rep_dim
        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        return self.code(x)


class MLP_Decoder(BaseNet):
    """MLP Decoder for autoencoder pretraining."""

    def __init__(self, x_dim, h_dims=[64, 128], rep_dim=32, bias=False):
        super().__init__()
        self.rep_dim = rep_dim
        neurons = [rep_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(layers)
        self.reconstruction = nn.Linear(h_dims[-1], x_dim, bias=bias)

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        return self.reconstruction(x)


class MLP_Autoencoder(BaseNet):
    """MLP Autoencoder for pretraining."""

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()
        self.rep_dim = rep_dim
        self.encoder = MLP(x_dim, h_dims, rep_dim, bias)
        self.decoder = MLP_Decoder(x_dim, list(reversed(h_dims)), rep_dim, bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class AETrainer:
    """Autoencoder Trainer for pretraining."""

    def __init__(self, lr: float = 0.001, n_epochs: int = 100, batch_size: int = 64, 
                 weight_decay: float = 1e-6, device: str = 'cpu'):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.train_time = None

    def train(self, train_loader, ae_net):
        logger = logging.getLogger()
        criterion = nn.MSELoss(reduction='mean')
        ae_net = ae_net.to(self.device)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        logger.info('Starting autoencoder pretraining...')
        start_time = time.time()
        ae_net.train()
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                rec = ae_net(inputs)
                loss = criterion(rec, inputs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0:
                logger.info(f'AE Epoch {epoch + 1:03}/{self.n_epochs:03} | Loss: {epoch_loss / n_batches:.6f}')

        self.train_time = time.time() - start_time
        logger.info(f'Pretraining finished in {self.train_time:.2f}s')
        return ae_net
    
class DeepSADTrainer:
    """Deep-SAD Trainer."""

    def __init__(self, c=None, eta: float = 1.0, lr: float = 0.001, n_epochs: int = 100, 
                 batch_size: int = 64, weight_decay: float = 1e-6, device: str = 'cpu'):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta
        self.eps = 1e-6
        self.train_time = None
        self.test_auc = None
        self.loss_history = []

    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center as mean of encoder outputs."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def train(self, train_loader, net):
        logger = logging.getLogger()
        net = net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)

        logger.info('Starting Deep-SAD training...')
        start_time = time.time()
        net.train()
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for data in train_loader:
                inputs, _, semi_targets, _ = data
                inputs = inputs.to(self.device)
                semi_targets = semi_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                
                # Deep-SAD loss: normal -> minimize dist, anomaly -> maximize dist
                losses = torch.where(
                    semi_targets == 0,
                    dist,
                    self.eta * ((dist + self.eps) ** semi_targets.float())
                )
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f'SAD Epoch {epoch + 1:03}/{self.n_epochs:03} | Loss: {avg_loss:.6f}')

        self.train_time = time.time() - start_time
        logger.info(f'Training finished in {self.train_time:.2f}s')
        return net

    def test(self, test_loader, net):
        logger = logging.getLogger()
        net = net.to(self.device)
        net.eval()
        
        idx_label_score = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, _, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                scores = torch.sum((outputs - self.c) ** 2, dim=1)
                
                idx_label_score += list(zip(
                    idx.cpu().numpy().tolist(),
                    labels.cpu().numpy().tolist(),
                    scores.cpu().numpy().tolist()
                ))

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)
        
        logger.info(f'Test AUC: {100. * self.test_auc:.2f}%')
        return labels, scores
