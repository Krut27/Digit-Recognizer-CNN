import numpy as np
import pandas as pd
import pickle

class Linear:
    # initialisng weight and biases
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, x): # for forward propogation
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, d_out): # for backward propogation
        self.grad_weights = np.dot(self.input.T, d_out)
        self.grad_biases = np.sum(d_out, axis=0, keepdims=True)
        return np.dot(d_out, self.weights.T)

# Activation classes

class ReLU: #Rectified Linear Unit activation
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_out): # gradients for ReLU activation
        d_input = d_out.copy()
        d_input[self.input <= 0] = 0
        return d_input

class Softmax: #softmax activation
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, d_out): # gradients for softmax activation
        return d_out

class CrossEntropyLoss: # calculating cross entropy loss
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.n = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-10)) / self.n

    def backward(self): # computing gradient of cross entropy loss
        return (self.y_pred - self.y_true) / self.n

# Stochastic Gradient Descent 
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
w
    def step(self, layer):
        if hasattr(layer, 'grad_weights'):
            layer.weights -= self.learning_rate * layer.grad_weights
            layer.biases -= self.learning_rate * layer.grad_biases

# Neural Network Model
class Model:
    def __init__(self): # initialising model
        self.layers = []
        self.loss_fn = None
        self.optimizer = None

    def add(self, layer): # adding layer to the neural network
        self.layers.append(layer)

    def compile(self, loss_fn, optimizer): # setting up loss and optimizer function
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, x): # forward pass
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad): # backward pass
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def train(self, x_train, y_train, epochs, batch_size): #TRAINING THE MODEL
        for epoch in range(epochs):
            indices = np.arange(x_train.shape[0])
            #np.random.shuffle(indices)
            x_train, y_train = x_train[indices], y_train[indices]

            for start in range(0, x_train.shape[0], batch_size):
                end = min(start + batch_size, x_train.shape[0])
                x_batch, y_batch = x_train[start:end], y_train[start:end]

                predictions = self.forward(x_batch)
                loss = self.loss_fn.forward(predictions, y_batch)
                loss_grad = self.loss_fn.backward()
                self.backward(loss_grad)

                for layer in self.layers:
                    self.optimizer.step(layer)

            print(f'EPOCH {epoch + 1}/{epochs}, LOSS: {loss}')

    def evaluate(self, x_test, y_test): # Evaluation of the model
        predictions = self.forward(x_test)
        loss = self.loss_fn.forward(predictions, y_test)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
        return loss, accuracy

    def save_weights(self, filename):
        # Save the weights and biases of the model to a file
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                weights_dict[f'layer_{i}_weights'] = layer.weights
                weights_dict[f'layer_{i}_biases'] = layer.biases
        
        with open(filename, 'wb') as f:
            pickle.dump(weights_dict, f)
        print(f"Model weights saved to {filename}")

    def load_weights(self, filename):
        # Loading the weights and biases from a file into the model
        with open(filename, 'rb') as f:
            weights_dict = pickle.load(f)
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                layer.weights = weights_dict[f'layer_{i}_weights']
                layer.biases = weights_dict[f'layer_{i}_biases']
        print(f"Model weights loaded from {filename}")

# Load and preprocess data
train_df = pd.read_csv('train.csv')
x = train_df.drop(columns='label').values / 255.0
y = pd.get_dummies(train_df['label']).values

# TO RUN THE MODEL::
# Define a simple neural network using the framework
model = Model()
model.add(Linear(784, 128))
model.add(ReLU())
model.add(Linear(128, 10))
model.add(Softmax())

# Compile the model with loss and optimizer
loss = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)
model.compile(loss, optimizer)

# Train the model
model.train(x, y, epochs=20, batch_size=64)

# LOAD OR SAVE WEIGHTS HERE

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x, y)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
