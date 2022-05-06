import torch.nn as nn
import torch as th

import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


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
        #print(self.layers)

        

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
  

class CNN(nn.Sequential):
    """CNN."""
    def __init__(self, in_size, out_size, net_arch,**kwargs):
        super(CNN,self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(4,8,kernel_size=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,8,kernel_size=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Linear(64, out_size)
        )

    def forward(self, input):
        return self.model(input)

class TargetDisc(nn.Module):
    def init(self, in_size, out_size,**kwargs):
        super(TargetDisc,self).__init__()
        conv_circ = nn.Conv1d(in_channels=2,out_channels=24,kernel_size=3,padding_mode='circular',padding='same')
    

class DiscRNN(nn.Module):

    def __init__(self, in_size, out_size, net_arch = [30,30], device="cpu", padding_idx=-1, gate_type="Rnn"):
        super().__init__()
        self.padding_idx = padding_idx
        self.nb_rnn_layers = len(net_arch)
        self.nb_rnn_units = net_arch[0]
        self.gate_type = gate_type
        self.disc_obs_shape = in_size
        self.device=device

        # don't count the padding tag for the classifier output
        self.out_size = out_size

        # when the model is bidirectional we double the output dimension


        # build actual NN
        self.__build_model()

    def __build_model(self):
        # build embedding layer first

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        #self.disc_embedding = nn.Embedding(
        #    num_embeddings=self.disc_obs_shape+1,
        #    embedding_dim=self.embedding_dim,
        #    padding_idx=self.padding_idx
        #)

        # design LSTM
        #self.lstm = nn.LSTM(
        #    input_size=self.embedding_dim,
        #    hidden_size=self.nb_rnn_units,
        #    num_layers=self.nb_rnn_layers,
        #    batch_first=True,
        #)

        if self.gate_type == "Rnn":
            self.rnn = nn.RNN(input_size=self.disc_obs_shape,
            hidden_size=self.nb_rnn_units,
            num_layers=self.nb_rnn_layers,
            batch_first=True,)

        elif self.gate_type == "Gru":
            print("using GRU")
            self.rnn = nn.GRU(input_size=self.disc_obs_shape,
            hidden_size=self.nb_rnn_units,
            num_layers=self.nb_rnn_layers,
            batch_first=True,)
        

        # output layer which projects back to tag space
        self.output_layer = nn.Linear(self.nb_rnn_units, self.out_size)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_rnn_units)
        hidden_a = th.randn(self.nb_rnn_layers, self.batch_size, self.nb_rnn_units).to(self.device)
        hidden_b = th.randn(self.nb_rnn_layers, self.batch_size, self.nb_rnn_units).to(self.device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return hidden_a

    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.batch_size, seq_len, _ = X.size()
        self.hidden = self.init_hidden()
        #print(self.hidden.shape)


        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        #X = self.disc_embedding(X)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_rnn_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X_lengths = X_lengths.to("cpu")
        X = th.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)

        # now run through LSTM
        X, self.hidden = self.rnn(X, self.hidden)

        # undo the packing operation
        X, _ = th.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, total_length=seq_len, padding_value=0)
        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_rnn_units) -> (batch_size * seq_len, nb_rnn_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        # run through actual linear layer
        X = self.output_layer(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_rnn_units) -> (batch_size, seq_len, nb_tags)
        X = X.view(self.batch_size, seq_len, self.out_size)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)

        Y_hat = X
        return Y_hat

    def loss(self, Y_hat, Y):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels

        mask = Y>self.padding_idx
        # count how many tokens we have
        nb_tokens = int(th.sum(mask[:,:,-1]).item())
        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -th.sum(Y_hat[mask]*Y[mask]) / nb_tokens
        return ce_loss



class Discriminator(nn.Module):

    """Estimate log p(z | s)."""

    def __init__(self, disc_obs_shape, out_size, net_arch, device = 'auto',arch_type='Mlp', optimizer_class = th.optim.Adam, lr = 0.0003, **kwargs):

        super(Discriminator, self).__init__()
        self.device = device
        self.arch_type = arch_type
        in_size = np.ravel(disc_obs_shape)[0]

        if arch_type=='Mlp':

            self.network = MLP(in_size, out_size, net_arch, **kwargs).to(self.device)

        elif arch_type=='Cnn':
            self.network = CNN(in_size, out_size, net_arch, **kwargs).to(self.device)

        elif arch_type=="Rnn":
            self.network = DiscRNN(in_size, out_size, net_arch,device=self.device, **kwargs).to(self.device)
        self.out_size = out_size
        self.optimizer = optimizer_class(self.parameters(), lr=lr)


    def forward(self, s, X_lengths=None):
        if not isinstance(s, th.Tensor):
            s = th.Tensor(s).to(self.device)
        if self.arch_type == "Rnn":
            a = self.network(s,X_lengths)
        else:
            a = self.network(s)
        if self.out_size == 1:
            #print(self.network(s).device)
            return th.log(th.sigmoid(a))
        return F.log_softmax(a, dim=-1)

    def loss(self, Y_hat, Y):
        if self.arch_type != "Rnn":
            return th.nn.NLLLoss()(Y_hat, Y.argmax(dim=1))
        
        else:
            return self.network.loss(Y_hat, Y)