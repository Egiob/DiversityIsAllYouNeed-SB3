import torch.nn as nn
import torch as th

import torch.nn.functional as F

class NeuralNetwork(nn.Sequential):

    """Fully-connected neural network."""

    def __init__(self, in_size, out_size, hidden_sizes, 
                 activation=nn.ReLU):
        super(NeuralNetwork, self).__init__()
        self.layers = []
        
        for size in hidden_sizes:
            self.layers.append(nn.Linear(in_size, size))
            self.layers.append(activation())
            in_size = size
        self.layers.append(nn.Linear(in_size, out_size))
        self.layers = nn.ModuleList(self.layers)

        

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
  
    


    @property
    def hidden_sizes(self):
        sizes = [
            c.in_features for c in self.children() if isinstance(c, nn.Linear)
        ]
        return sizes[1:]

    @property
    def in_size(self):
        sizes = [
            c.in_features for c in self.children() if isinstance(c, nn.Linear)
        ]
        return sizes[0]

    @property
    def out_size(self):
        sizes = [
            c.out_features for c in self.children() if isinstance(c, nn.Linear)
        ]
        return sizes[-1]


class Discriminator(nn.Module):

    """Estimate log p(z | s)."""
    def __init__(self, disc_obs_shape, prior, hidden_sizes, device = 'auto', **kwargs):
        
        super(Discriminator, self).__init__()
        self.device = device
        in_size = disc_obs_shape
        out_size = prior.param_shape[0] if prior.param_shape else 1
        self.network = NeuralNetwork(in_size, out_size, hidden_sizes, **kwargs).to(self.device)
        self.out_size = out_size
        self.optimizer = th.optim.Adam(self.parameters())

    def forward(self, s):
        if not isinstance(s, th.Tensor):
            s = th.Tensor(s).to(self.device)
        if self.out_size == 1:
            #print(self.network(s).device)
            return th.log(th.sigmoid(self.network(s)))
        return F.log_softmax(self.network(s), dim=1)