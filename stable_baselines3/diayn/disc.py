import torch.nn as nn
import torch as th

import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d

class MLP(nn.Sequential):

    """Fully-connected neural network."""

    def __init__(self, in_size, out_size, hidden_sizes, 
                 activation=nn.ReLU, **kwargs):
        super(MLP, self).__init__()
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
  

class CNN(nn.Sequential):
    """CNN."""
    def __init__(self, in_size, out_size, net_arch,**kwargs):
        super(CNN,self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3,8,kernel_size=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,8,kernel_size=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Linear(32, out_size)
        )

    def forward(self, input):
        return self.model(input)




class Discriminator(nn.Module):

    """Estimate log p(z | s)."""

    def __init__(self, disc_obs_shape, prior, net_arch, device = 'auto',arch_type='Mlp', optimizer_class = th.optim.Adam, lr = 0.0001, **kwargs):

        super(Discriminator, self).__init__()
        self.device = device
        in_size = disc_obs_shape
        out_size = prior.param_shape[0] if prior.param_shape else 1
        if arch_type=='Mlp':

            self.network = MLP(in_size, out_size, net_arch, **kwargs).to(self.device)

        elif arch_type=='Cnn':
            self.network = CNN(in_size, out_size, net_arch, **kwargs).to(self.device)

        self.out_size = out_size
        self.optimizer = optimizer_class(self.parameters(), lr=lr)


    def forward(self, s):
        if not isinstance(s, th.Tensor):
            s = th.Tensor(s).to(self.device)
        if self.out_size == 1:
            #print(self.network(s).device)
            return th.log(th.sigmoid(self.network(s)))
        return F.log_softmax(self.network(s), dim=1)