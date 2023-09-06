import numpy as np


fred = np.array([1, 2, 3])
greg = np.transpose(fred)

print(fred)
print(greg)

grebb = np.array([[1, 2], [1, 2]])
print(grebb)
glogg = np.transpose(grebb)
print(grebb)
print(glogg)

print("flubb\n")
print(grebb.T)

print(fred.T)

ja = [1, 1]
ju = np.dot(ja, grebb)
print(ju)

hus = np.zeros((12, 3))
print(hus)



E = np.zeros((18, 2))
vector = np.ones(18)

E[:, 0] += vector

print(E)

print("~~~~~~")
print()

E = np.zeros((3, 5))
vector = np.ones((3, 2))
E[:, 3:] += vector
print(E)

print()


# Create a 2D array (matrix) with shape (18, 5)
matrix = np.zeros((18, 5))

# Create a 1D array (vector) with 18 elements
vector = np.arange(18)

# Add the vector to the first and second columns of the matrix
matrix[:, 0:2] += vector[:, None]

print(matrix)

print()
print("~~~~~~")
greggy = np.array([i for i in range(5)])
print(greggy[:, None])

import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
gobby = torch.zeros(5)
print(gobby)
print(gobby.device.type)

golgo = torch.zeros((5, 2))
print(golgo)
print(golgo.T)

E = torch.zeros((18, 2))
klubb = torch.ones(18)
E[:, 0] += klubb
print(E)


jubba = torch.randint(10, (1,)).item()
print(jubba.type)