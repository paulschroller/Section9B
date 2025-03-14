import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from numpy.lib.stride_tricks import sliding_window_view

#Task 1
class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        delta = (self.activations[-1] - y) / m
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.z_values[i-1])
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

class CNN:
    def __init__(self, input_shape, num_filters, filter_size, num_classes):
        self.input_shape = input_shape  # (height, width)
        self.num_filters = num_filters  # number of filters
        self.filter_size = filter_size  # size of each filter (e.g., 3x3)
        self.num_classes = num_classes  # number of output classes
        # Initialize filters and biases
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.01
        self.biases = np.zeros((num_filters, 1))

        # Calculate the output size after convolution
        self.output_size = input_shape[0] - filter_size + 1

        # Initialize fully connected layer (using MLP from Task 1)
        # Input size to MLP: num_filters * output_size * output_size
        mlp_input_size = num_filters * self.output_size * self.output_size
        self.mlp = MLP([mlp_input_size, 128, num_classes])
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def conv2d_vectorized(self, X, filters):
        n_filters, filter_size, _ = filters.shape
        batch_size, input_size, _ = X.shape
        output_size = input_size - filter_size + 1

        # Create sliding windows for the input
        X_windows = sliding_window_view(X, (1, filter_size, filter_size))
        X_windows = X_windows.reshape(batch_size, output_size, output_size, -1)

        # Reshape filters to match the dimensions of X_windows
        filters_reshaped = filters.reshape(n_filters, -1)

        # Perform convolution using einsum
        output = np.einsum('bhwc,fc->bhwf', X_windows, filters_reshaped)
        return output.transpose(0, 3, 1, 2)  # Reshape to (batch_size, n_filters, output_size, output_size)

    def forward(self, X):
        # Convolutional layer
        self.conv_output = self.conv2d_vectorized(X, self.filters)
        self.conv_output = self.sigmoid(self.conv_output)

        # Flatten the output for the fully connected layer
        self.flattened = self.conv_output.reshape(X.shape[0], -1)
        return self.mlp.forward(self.flattened)

    def backward_vectorized(self, X, y, learning_rate=0.01):
        # Backprop through MLP
        self.mlp.backward(self.flattened, y, learning_rate)

        # Backprop through Conv layer
        delta = self.mlp.activations[-1] - y
        for i in reversed(range(len(self.mlp.weights))):
            delta = np.dot(delta, self.mlp.weights[i].T)
            if i > 0:
                delta *= self.sigmoid_derivative(self.mlp.z_values[i-1])

        # Reshape delta to match conv_output shape
        delta = delta.reshape(self.conv_output.shape)

        # Update filters and biases using vectorized operations
        X_windows = sliding_window_view(X, (1, self.filter_size, self.filter_size))
        X_windows = X_windows.reshape(X.shape[0], self.output_size, self.output_size, -1)

        for i in range(self.num_filters):
            grad_filter = np.einsum('bhwc,bhw->c', X_windows, delta[:, i])
            self.filters[i] -= learning_rate * grad_filter.reshape(self.filter_size, self.filter_size)
            self.biases[i] -= learning_rate * np.sum(delta[:, i])

    def train(self, X, y, epochs, learning_rate, batch_size):
        num_batches = X.shape[0] // batch_size
        for epoch in range(epochs):
            for batch in range(num_batches):
                X_batch = X[batch * batch_size:(batch + 1) * batch_size]
                y_batch = y[batch * batch_size:(batch + 1) * batch_size]
                self.forward(X_batch)
                self.backward_vectorized(X_batch, y_batch, learning_rate)



mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot encoding
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

y_train_one_hot = one_hot_encode(y_train, 10)
y_test_one_hot = one_hot_encode(y_test, 10)


# Train MLP
mlp = MLP([784, 128, 64, 10])
mlp.train(X_train, y_train_one_hot, epochs=200, learning_rate=1)

# Evaluate MLP
y_pred_mlp = mlp.forward(X_test)
y_pred_mlp = np.argmax(y_pred_mlp, axis=1)
print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("MLP Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp))
# Plot confusion matrix

def plot_confusion_matrix(cm, title, directory):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.savefig(directory)
    plt.show()

plot_confusion_matrix(confusion_matrix(y_test, y_pred_mlp), "MLP Confusion Matrix", "./plots/MLPconfuion")


# Convert X_train and X_test to NumPy arrays and reshape for CNN
X_train_cnn = X_train.to_numpy().reshape(-1, 28, 28)
X_test_cnn = X_test.to_numpy().reshape(-1, 28, 28)

# Train CNN
cnn = CNN(input_shape=(28, 28), num_filters=1, filter_size=3, num_classes=10)
cnn.train(X_train_cnn, y_train_one_hot, epochs=20, learning_rate=0.01, batch_size=8)

y_pred_cnn = []
for i in range(len(X_test_cnn)):
    y_pred_cnn.append(cnn.forward(X_test_cnn[i:i+1]))  # Forward pass for one example at a time
y_pred_cnn = np.argmax(np.vstack(y_pred_cnn), axis=1)
print("CNN Accuracy:", accuracy_score(y_test, y_pred_cnn))
print("CNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_cnn))

plot_confusion_matrix(confusion_matrix(y_test, y_pred_cnn), "CNN Confusion Matrix", "./plots/CNNconfusion")