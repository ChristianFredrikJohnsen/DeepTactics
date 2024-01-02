import pickle
import gzip
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from icecream import ic

def load_data():

    f = gzip.open('Neural_Architectures/basic_dense_torch/data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding = 'latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    
    training_results = [vectorized_result(y) for y in tr_d[1]]
    
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]

    validation_data = list(zip(validation_inputs, va_d[1]))
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)


def load_data_wrapper_torch(device) -> tuple[DataLoader, list[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:

    training_data, validation_data, test_data = load_data_wrapper() # Your existing method

    # Convert data to PyTorch tensors and then to a Dataset
    train_x = [np.reshape(x, (784, )) for x, _ in training_data]  # Flatten the input
    train_y = [np.reshape(y, (10, ))  for _ , y in training_data]  # Extract labels

    # Convert to tensors, setting up for GPU usage
    tensor_x = torch.stack([torch.Tensor(inp).to(device) for inp in train_x])
    tensor_y = torch.stack([torch.Tensor(output).to(device) for output in train_y])
    ic(tensor_x.shape)
    ic(tensor_y.shape)

    alt = (tensor_x, tensor_y)

    """
    It is way faster to load the entire dataset into the GPU memory, rather than loading it in batches.
    """
    # mnist_dataset = TensorDataset(tensor_x, tensor_y)
    # mnist_dataloader = DataLoader(mnist_dataset, batch_size = 50_000, shuffle=True)

    validation_data = torch_conversion(validation_data, device) # shape: (10000, 784), (10000, )
    test_data = torch_conversion(test_data, device) # shape: (10000, 784), (10000, )
    ic(test_data[1].shape)
    ic(tensor_x.shape)
    return (alt, validation_data, test_data)



def vectorized_result(j: int):
    
    # With my own words, i would say that this creates a column vector which represents the target for our network.
    # Our network has 10 outputs, so the target for the network must be a 10D column vector.
    # We set the target of the network to be 0's for all numbers except the correct number, which should be labelled as 1.

    return np.array([[int(i == j)] for i in range(10)])

def torch_conversion(dataset: list[tuple[torch.Tensor, torch.Tensor]], device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts data on the format (x, y) to a tuple of tensors (tensor_x, tensor_y).
    So instead of a long list of tuples, we have some large tensors representing input and wanted output.
    """

    data_x = [np.reshape(x, (784, )) for x, _ in dataset]  # Flatten the input
    
    tensor_x = torch.stack([torch.Tensor(arg).to(device) for arg in data_x])
    tensor_y = torch.tensor([output for _, output in dataset]).to(device)

    return (tensor_x, tensor_y)

def main():

    # Load the MNIST data
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train, valid, test = load_data_wrapper_torch(dev)
    ic(train[1][0])



    
if __name__ == "__main__":
    main()
