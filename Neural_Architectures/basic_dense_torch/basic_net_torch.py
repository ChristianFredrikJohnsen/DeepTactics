from icecream import ic
import mnist_loader_torch
import cProfile
import torch
from sigmoid_activation import quick_sigmoid, sigmoid
from mse_loss import mse_prime
from data_visualizer_mnist import visualize_sample, visualize_samples

class Network:

    def __init__(self, sizes: list[int]):
        
        self.num_layers: int = len(sizes)
        """
        The number of layers in the network. Includes input and output layers.
        """

        self.sizes = sizes
        """
        The number of neurons in each layer.\n
        [2, 3, 1] will give you an input layer of size 2, hidden layer of size 3, and output layer of size 1.
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        """
        GPU acceleration is used if available.
        """

        self.biases: list[torch.Tensor] = [torch.randn(y, 1, device = self.device, dtype = torch.float32) for y in sizes[1:]]
        """
        A list of column vectors, where each vector contains the biases for a layer.\n
        If you have a net with shape [784, 30, 10], then the first vector will have a shape of (30, 1).
        """

        self.weights: list[torch.Tensor] = [torch.randn(y, x, device = self.device, dtype = torch.float32) for x, y in zip(sizes[:-1], sizes[1:])]
        """
        A list of matrices, where each matrix contains the weights between two layers.\n
        If you have a net with shape [784, 30, 10], then the first matrix will have a shape of (30, 784).
        """

        self.nabla_b = [torch.zeros_like(b) for b in self.biases]
        """
        Matrix representing the gradient of the cost function with respect to the biases for a single mini-batch.
        """

        self.nabla_w = [torch.zeros_like(w) for w in self.weights]
        """
        Matrix representing the gradient of the cost function with respect to the weights for a single mini-batch.
        """

    def feedforward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Returns the output of the network, provided that a is the input.
        It is important to note that this method expects a (n, batch_size) array as input, not a (n,) vector.\n
        Returns a (num_outputs, batch_size) array, representing the predictions for each training example in the mini-batch.
        """
        for w, b in zip(self.weights, self.biases): # For each layer
            a = sigmoid(w @ a + b) # Calculate all of the activations for the next layer.
        
        return a 
    
    def SGD(self, epochs: int, alpha: float, batch_size: int,
            training_data: tuple[torch.Tensor, torch.Tensor],
            test_data: tuple[torch.Tensor, torch.Tensor] = None):
        """
        Training the neural network using the stochastic gradient descent algorithm.
        Training data is a tuple (x, y) representing what the input is and what the desired output is.

        X shape: (batch_size, num_inputs) - num_inputs = 784 in the case of MNIST
        Y shape: (batch_size, num_outputs) - num_outputs = 10 in the case of MNIST
        """
        
        x, y = training_data[0], training_data[1]
        test_num = test_data[0].size(0) if test_data else 0 # Number of test examples.
        n = x.size(0) # Number of training examples.

        for epoch in range(1, epochs + 1):
            
            indices = torch.randperm(n) # Randomize the indices of the training examples.
            
            for i in range(0, n, batch_size):
                batch_indices = indices[i:i+batch_size] # Get the indices of the training examples in the mini-batch.
                self.update_mini_batch(x[batch_indices], y[batch_indices], alpha)

            if (epoch % 10 == 0 or epoch == epochs):

                if test_data:
                    print(f'Epoch {epoch}: {self.evaluate(test_data)} / {test_num}')
                
                else:
                    print(f'Epoch {epoch} complete')
    
    def update_mini_batch(self, x: torch.Tensor, y: torch.Tensor, alpha: float):
        """
        This is the part where we are actually updating the network's weights and biases with the
        stochastic gradient descent algorithm. We find the gradient of the loss function for a single mini-batch,
        and then we update the weights and biases accordingly.

        X shape: (batch_size, 784)
        Y shape: (batch_size, 10)
        """

        self.backprop(x, y) # Updates the nabla_b and nabla_w matrices for a single batch of training examples.
        
        for w, b, nw, nb in zip(self.weights, self.biases, self.nabla_w, self.nabla_b):
            w -= alpha * nw # Update the weights based on the gradient of the loss-function.
            b -= alpha * nb # Update the biases as well.
    
    def backprop(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Does not return anything, but updates the nabla_b and nabla_w matrices.
        VERY IMPORTANT: The whole purpose of this method is to find the gradient of the loss function for the mini-batch.

        X shape: (batch_size, 784)
        Y shape: (batch_size, 10)
        """
        sps, activations = self.forward_pass(x.T) # Do a forward pass and store all information about σ'(z) and activations.
        self.backward_pass(activations, sps, y, x.size(0)) # Do a backward pass, using the information from the forward pass, and find the gradient.

    def forward_pass(self, a: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        A method used during backpropagation, which not only does inference, but also stores the intermediate values 
        (activations and derivatives of weighted inputs), which are needed to calculate the gradient.\n
        X shape: (784, batch_size)\n
        Returns: (sigmoid_primes, activations), Shapes: (num_neurons, batch_size)
        """
        sps = []; activations = [a]

        for b, w in zip(self.biases, self.weights):
            z = w @ a + b # Calculate weighted sum of inputs for each neuron in the layer.
            a, sp = quick_sigmoid(z) # a = σ(z), sp = σ'(z)
            sps.append(sp); activations.append(a) # Store the information from this layer.

        return (sps, activations)

    def backward_pass(self, activations: list[torch.Tensor], sps: list[torch.Tensor], y: torch.Tensor, batch_size: int) -> None:
        """
        This method does not return anything, but updates the nabla_b and nabla_w matrices to contain the gradient for the mini-batch.\n
        activations and sps shape: (num_neurons, batch_size) for each layer.\n
        y-shape: (batch_size, num_outputs) - num_outputs = 10 in the case of MNIST
        """

        delta = mse_prime(y.T, activations[-1]) * sps[-1] # Calculate dL/dz (delta) for the output layer. Shape = (num_neurons, batch_size)
        self.get_gradient(delta, activations, 1, batch_size) # Update the nabla_b and nabla_w matrices for the output layer.
        
        for l in range(2, self.num_layers): # Go from layer (output_layer - 1) to layer 1.
            
            delta = self.weights[-l + 1].T @ delta * sps[-l] # Calculate deltas, should have shape (num_neurons, batch_size). One delta vector for each training example in the mini-batch.
            self.get_gradient(delta, activations, l, batch_size) # Update the nabla_b and nabla_w matrices for layer (num_layers - l).
    
    def get_gradient(self, delta: torch.Tensor, activations: torch.Tensor, layer: int, batch_size: int) -> None:
        """
        Uses the value of delta (dL/dz) to calculate the gradient of the loss function with respect to the weights and biases in layer l.\n
        
        Parameters:\n
        - Delta; shape: (num_neurons_l , batch_size)\n
        - Activations; shape: (num_neurons_l-1, batch_size)\n
        - Layer; layers are one-indexed with the output layer being layer 1.\n
        
        Updates: nabla_b and nabla_w | shapes: (num_neurons_l, 1), (num_neurons_l, num_neurons_l-1)\n

        Nabla_b is the gradient of the loss function with respect to the biases in layer l,
        and nabla_w is the gradient of the loss function with respect to the weights going from layer l-1 to layer l.\n
        Of course, this is not the true gradient (which would be the average gradient over all training examples), but rather the gradient over a single mini-batch.
        """
        self.nabla_w[-layer] = delta @ activations[-layer-1].T / batch_size
        self.nabla_b[-layer] = torch.mean(delta, dim = 1, keepdim = True) # Calculate the mean over the batch_size dimension, and keep the column dimension.

    def evaluate(self, test_data: tuple[torch.Tensor, torch.Tensor]) -> int:
        """
        Returns the number of correct predictions for the test data.
        The prediction is assumed to be the argmax of output layer.

        Assuming test data has shapes (10000, 784), (10000, )
        """
        x, y = test_data[0], test_data[1]
        predictions = self.feedforward(x.T).argmax(dim=0) # Get a tensor with shape (10000,), one prediction for each test example.
        return torch.sum(predictions == y).item() # Sum up the number of correct predictions. item() is used to convert the tensor to a python int.

    def save(self, filepath: str) -> None:
        """
        Saves the network's parameters to a file.
        """
        torch.save({'weights': self.weights, 'biases': self.biases}, filepath)

    def load(self, filepath: str) -> None:
        """
        Loads the network's parameters from a file.\n
        The loading method works even if you don't have a GPU with CUDA support.
        """
        parameters = torch.load(filepath, map_location = self.device)
        self.weights = parameters['weights']
        self.biases = parameters['biases']

    @property
    def num_params(self) -> int:
        """
        Returns the number of parameters in the network.
        """
        return sum(w.size().numel() + b.size().numel() for w, b in zip(self.weights, self.biases))

def training(net: Network, epochs: int, learning_rate: float, batch_size: int, training_data, test_data = None):
    """
    Method used to train and save a network.\n
    If you for some reason want to abort the training process, you can press CTRL + C, and the network will be saved.
    """
    try:
        net.SGD(epochs, learning_rate, batch_size, training_data, test_data = test_data) # epochs, learning rate, batch size
        print("\nSaving!")
        net.save(filename)

    except KeyboardInterrupt:
        print("\nSaving!")
        net.save(filename)

if __name__ == '__main__':

    ### Filepath used for saving/loading of network.
    prefix = 'Neural_Architectures/basic_dense_torch/trained_networks/'
    file = '784_30_10.pyt'
    file2 = '784_30_10V2.pyt'
    filename = prefix + file
    trained_file = prefix + file2

    ### Create a custom network, important part is that input layer has 784 neurons and output layer has 10 neurons.
    net = Network([784, 30, 10])
    ic(net.num_params)

    ### Load the MNIST dataset. If your network is using the GPU, then the data will be loaded into the GPU memory.
    training_data, validation_data, test_data = mnist_loader_torch.load_data_wrapper_torch(net.device)

    ### Train the network and save it.
    # training(net, epochs = 30, learning_rate = 2, batch_size = 100, training_data = training_data, test_data = test_data)

    ### Load some other trained network and do inference with it.  
    net.load(trained_file)
    train1 = training_data[0][0].unsqueeze(1)
    ic(net.feedforward(train1)) # The network should predict a 5. To get the prediction, take the index of the largest value in the output layer.
    
    # visualize_sample(net, test_data)
    visualize_samples(net, test_data, start_index = 6421, num_samples = 10)


    ### Use cProfile to profile the SGD function for one epoch. This is done to see if there are any bottlenecks.
    # cProfile.run('net.SGD(epochs = 1, alpha = 2, batch_size = 100, training_data = training_data, test_data = test_data)')


"""
Insane optimization: Instead of calculating the gradient for each training example in the mini-batch separately,
we can calculate the gradient for all of them at once. This is done by setting up one matrix multiplication representing a
computation of the forward pass for all of the training examples in the mini-batch.
The GPU is optimized for matrix multiplication, so this is a huge speedup.

The lists sps and activations both have shape (num_neurons, batch_size) for each layer when the loop is done.
For example, in the case of the activation list, the first entry holds a tensor of size (784, batch_size),
which in practice gives you the activations of the first layer for all of the training examples in the mini-batch.

This was a comment on the forward_pass method, where we are able to do inference on all of the training examples in the mini-batch in one mathematical operation,
with the help of clever matrix multiplication.

The calculation: z = w @ a + b
may seem simple, but if you have large batch sizes, this computation becomes really expensive.
For example, you may have a batch size of 64, and if you are doing the first (with second layer having 30 neurons), then you will need to compute the result of a
matrix multiplication between a (30, 784) matrix and a (784, 64) matrix.

If you do the "normal" matrix multiplication algorithm with O(n^3) complexity, then you will need to do 30 * 784 * 64 = 1 505 280 multiplications, which is a lot.
For the GPU however, such sizes are trivial.
As a sidenote, the algorithms used for matrix multiplication are really optimized, and run in something like O(n^2.4) time on square matrices.
"""