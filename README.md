# Neural Network
## **DIGIT RECOGNIZER**

This repository contains a simple neural network framework implemented in Python to recognize various handwritten digits.
<br>
 The framework includes classes for linear layers, activation functions (ReLU and Softmax), loss function (Cross-Entropy), and an optimizer (Stochastic Gradient Descent).

 **ACCURACY: ~ 93.2%**<br>
 **LOSS: 0.237**<br>
 <br>Train it more to get better results!


## Requirements

- NumPy
- Pandas
- Pickle

## Usage

1. **Import the necessary classes:**

```python
from framework import Model, Linear, ReLU, Softmax, CrossEntropyLoss, SGD
```

2. **Create a model:**

```python
model = Model()
model.add(Linear(784, 128))
model.add(ReLU())
model.add(Linear(128, 10))
model.add(Softmax())
```

3. **Compile the model:**

```python
loss = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)
model.compile(loss, optimizer)
```

4. **Train the model:**

```python
model.train(x_train, y_train, epochs=20, batch_size=64)
```
NOTE: You can also load your already adjusted weights into the model. Similarly, You can also save the weights once you train the model. We'll look into it in a while.

5. **Evaluate the model:**

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
```

## Saving and Loading Weights

To save the model weights: ( after training )

```python
model.save_weights('model_weights.pkl')
```

To load the model weights: ( before training )

```python
model.load_weights('model_weights.pkl')
```

## Kaggle Notebook

For a better implementation of how to use this framework, please refer to my Kaggle notebook:

[CLICK HERE!](https://www.kaggle.com/code/krutikrana/neural-network)
