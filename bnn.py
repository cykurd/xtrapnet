import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

class BayesianXtrapNet(PyroModule):
    def __init__(self, input_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_dim, 64)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([64, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([64]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](64, 64)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([64, 64]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([64]).to_event(1))
        self.out = PyroModule[nn.Linear](64, 1)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([1, 64]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, y=None):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        mean = self.out(x).squeeze(-1)
        sigma = pyro.sample("sigma", dist.Uniform(0., 5.))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean

class BayesianTrainer:
    def __init__(self, network, learning_rate=0.01, num_epochs=1000, batch_size=100):
        self.network = network
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.guide = pyro.infer.autoguide.AutoNormal(self.network)
        self.svi = SVI(self.network, self.guide, Adam({"lr": self.lr}), Trace_ELBO())

    def train(self, labels, features):
        dataset = torch.utils.data.TensorDataset(torch.tensor(features, dtype=torch.float32),
                                                 torch.tensor(labels.squeeze(), dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.num_epochs):
            for x, y in loader:
                self.svi.step(x, y)

    def predict(self, features, n_samples=100):
        x = torch.tensor(features, dtype=torch.float32)
        preds = []
        for _ in range(n_samples):
            sampled_model = pyro.poutine.trace(self.guide).get_trace(x)
            preds.append(sampled_model.nodes["obs"]["fn"].loc.detach())
        preds = torch.stack(preds)
        return preds.mean(dim=0).unsqueeze(-1).numpy(), preds.var(dim=0).unsqueeze(-1).numpy()
