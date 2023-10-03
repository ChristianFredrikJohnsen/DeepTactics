from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
# Reshape the images to have one channel (grayscale) and normalize the pixel values
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the CNN model
model = Sequential()

# Convolutional layer: 32 filters of size 3x3, ReLU activation
# This layer will learn 32 different types of features (like edges, corners, etc.)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Max-Pooling layer: reduces the spatial dimensions (makes the detection of features invariant to scale and orientation changes)
model.add(MaxPooling2D((2, 2)))

# Flatten layer: flattens the 3D output to 1D tensor
model.add(Flatten())

# Fully connected layer: 64 neurons, ReLU activation
# This layer will learn to combine the features learned by the conv layer to form higher-level features
model.add(Dense(64, activation='relu'))

# Output layer: 10 neurons (one for each digit), softmax activation
# This layer will output the probability score for each class (digit)
model.add(Dense(10, activation='softmax'))

# Compile the model
# We use categorical_crossentropy as the loss function (good for multi-class classification problems)
# and Adam as the optimizer.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
model.summary()
