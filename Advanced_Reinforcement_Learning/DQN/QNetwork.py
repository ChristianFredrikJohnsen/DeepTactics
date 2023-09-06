import torch.nn as nn
import torch


class QNetwork(nn.Module):

    def __init__(self, ob_dim, action_space, hidden_dim=25):
        
        super().__init__()
        self.linear1 = nn.Linear(ob_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_space)

        self.activation = nn.ReLU()

    def forward(self, x):
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x
    
    


def num_params(module):
    return sum(param.numel() for param in module.parameters())


if __name__ == '__main__':
    network = QNetwork(4, 2)
    print(num_params(network))
    print(network(torch.zeros(4)))
