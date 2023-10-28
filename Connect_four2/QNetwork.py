import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, object_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        
        self.lin1 = nn.Linear(object_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, action_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.activation(x)
        x = self.lin3(x)
        return x
    
if __name__ == '__main__':
    net = QNetwork(42, 7, 500)
    print(f"Number of parameters: {sum(p.numel() for p in net.parameters())}") # Check the number of parameters.