import pickle
import gzip
from torch.utils.data import DataLoader, TensorDataset
import torch
from icecream import ic

def load_data():

    f = gzip.open('Neural_Architectures/basic_dense_torch/data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding = 'latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper_torch(device) -> tuple[DataLoader, list[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
    
    training_data, validation_data, test_data = load_data()

    train_x = [x for x in training_data[0]]
    train_y = [vectorized_result(y) for y in training_data[1]] 

    tensor_x = torch.stack([torch.Tensor(inp).to(device) for inp in train_x]) # shape: (50000, 784)
    tensor_y = torch.stack([torch.Tensor(output).to(device) for output in train_y]) # shape: (50000, 10)

    alt = (tensor_x, tensor_y)

    """
    It is way faster to load the entire dataset into the GPU memory, rather than loading it in batches.
    """
    # mnist_dataset = TensorDataset(tensor_x, tensor_y)
    # mnist_dataloader = DataLoader(mnist_dataset, batch_size = 50_000, shuffle=True)
    
    validation_data = validation_conversion(validation_data, device) # shape: (10000, 784), (10000, )
    test_data = validation_conversion(test_data, device) # shape: (10000, 784), (10000, )
    return (alt, validation_data, test_data)

def vectorized_result(j: int):
    
    # With my own words, i would say that this creates a column vector which represents the target for our network.
    # Our network has 10 outputs, so the target for the network must be a 10D column vector.
    # We set the target of the network to be 0's for all numbers except the correct number, which should be labelled as 1.

    return torch.tensor([int(i == j) for i in range(10)])

def validation_conversion(dataset: list[tuple[torch.Tensor, torch.Tensor]], device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts data on the format (x, y) to a tuple of tensors (tensor_x, tensor_y).
    So instead of a long list of tuples, we have some large tensors representing input and wanted output.
    """
    tensor_x = torch.stack([torch.Tensor(inp).to(device) for inp in dataset[0]])
    tensor_y = torch.tensor([output for output in dataset[1]]).to(device)
    return (tensor_x, tensor_y)

if __name__ == "__main__":

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train, valid, test = load_data_wrapper_torch(dev)
    ic(train[1][0])