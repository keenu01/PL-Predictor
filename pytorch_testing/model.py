import torch as t
import torch.nn as nn
import torch.nn.functional as f


class NeuralNetwork(nn.Module):
    def __init__(self,input_size=5,hidden_layers=50,output_size=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size,hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers,hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers,hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers,hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers,output_size),
                                 nn.Sigmoid()
                                 )
    def forward(self,x):
            return self.net(x)