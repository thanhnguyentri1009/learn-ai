# Neural Networks

## Perceptron - Building Block của Neural Networks

**Perceptron** là đơn vị cơ bản nhất của neural network, lấy cảm hứng từ tế bào thần kinh sinh học.

```
Inputs:  x1, x2, x3
Weights: w1, w2, w3
Bias:    b

         x1 * w1 ─┐
         x2 * w2 ─┤──> Σ ──> activation(z) ──> output
         x3 * w3 ─┘      z = w1*x1 + w2*x2 + w3*x3 + b
```

```python
import numpy as np

class Perceptron:
    """Perceptron đơn giản nhất."""

    def __init__(self, n_features: int, learning_rate: float = 0.01):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.lr = learning_rate

    def activation(self, z: float) -> int:
        """Step function: output 1 nếu z >= 0, ngược lại 0."""
        return 1 if z >= 0 else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return np.array([self.activation(zi) for zi in z])

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Perceptron learning rule: cập nhật khi dự đoán sai."""
        history = []
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                z = xi @ self.weights + self.bias
                y_pred = self.activation(z)
                error = yi - y_pred

                # Update rule
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                errors += int(error != 0)

            history.append(errors)
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

        return history

# Test với AND gate
X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
y_and = np.array([0, 0, 0, 1])

p = Perceptron(n_features=2, learning_rate=0.1)
p.fit(X_and, y_and, epochs=100)
print(f"Predictions: {p.predict(X_and)}")   # [0, 0, 0, 1]
print(f"Weights: {p.weights}, Bias: {p.bias:.2f}")
```

## Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid: maps any value to (0, 1). Dùng cho binary classification output."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    """ReLU: max(0, z). Phổ biến nhất cho hidden layers."""
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU: giải quyết 'dying ReLU' problem."""
    return np.where(z > 0, z, alpha * z)

def tanh(z):
    """Tanh: maps to (-1, 1). Cũ hơn ReLU nhưng vẫn dùng."""
    return np.tanh(z)

def softmax(z):
    """Softmax: chuyển vector thành probability distribution. Dùng cho multi-class output."""
    exp_z = np.exp(z - z.max())  # Numerical stability
    return exp_z / exp_z.sum(axis=-1, keepdims=True)

def gelu(z):
    """GELU: Dùng trong Transformers (BERT, GPT)."""
    return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

# Visualize
z = np.linspace(-5, 5, 200)
activations = {
    'Sigmoid': sigmoid(z),
    'ReLU': relu(z),
    'Leaky ReLU': leaky_relu(z),
    'Tanh': tanh(z),
    'GELU': gelu(z),
}

fig, axes = plt.subplots(1, len(activations), figsize=(20, 4))
for ax, (name, values) in zip(axes, activations.items()):
    ax.plot(z, values, 'b-', linewidth=2)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(name)
    ax.grid(True, alpha=0.3)

plt.suptitle('Activation Functions', fontsize=14)
plt.tight_layout()
plt.show()
```

## Feedforward Neural Network từ đầu

```python
import numpy as np
from typing import List, Tuple

class Layer:
    """Một layer trong neural network."""

    def __init__(self, n_inputs: int, n_neurons: int, activation: str = 'relu'):
        # He initialization cho ReLU, Xavier cho sigmoid/tanh
        if activation == 'relu':
            scale = np.sqrt(2.0 / n_inputs)
        else:
            scale = np.sqrt(1.0 / n_inputs)

        self.weights = np.random.randn(n_inputs, n_neurons) * scale
        self.biases = np.zeros((1, n_neurons))
        self.activation_name = activation

        # Cache cho backpropagation
        self.input = None
        self.z = None
        self.output = None
        self.dW = None
        self.db = None

    def activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation_name == 'relu':
            return np.maximum(0, z)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        elif self.activation_name == 'softmax':
            exp_z = np.exp(z - z.max(axis=1, keepdims=True))
            return exp_z / exp_z.sum(axis=1, keepdims=True)
        elif self.activation_name == 'linear':
            return z
        return z

    def activate_derivative(self, z: np.ndarray) -> np.ndarray:
        if self.activation_name == 'relu':
            return (z > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(z) ** 2
        return np.ones_like(z)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        self.z = X @ self.weights + self.biases
        self.output = self.activate(self.z)
        return self.output

    def backward(self, delta: np.ndarray) -> np.ndarray:
        n = self.input.shape[0]
        if self.activation_name not in ['softmax', 'linear']:
            delta = delta * self.activate_derivative(self.z)

        self.dW = self.input.T @ delta / n
        self.db = delta.mean(axis=0, keepdims=True)

        return delta @ self.weights.T

class NeuralNetwork:
    """
    Feedforward Neural Network với backpropagation.
    Hỗ trợ: binary classification, multi-class classification, regression.
    """

    def __init__(self, layer_sizes: List[int], activations: List[str]):
        """
        Args:
            layer_sizes: [n_input, n_hidden1, n_hidden2, ..., n_output]
            activations: activation function cho mỗi layer (trừ input)
        """
        assert len(activations) == len(layer_sizes) - 1

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)

        self.train_losses = []
        self.val_losses = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, task: str) -> float:
        """Tính loss dựa trên task."""
        n = y_true.shape[0]
        eps = 1e-15  # Tránh log(0)

        if task == 'binary':
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        elif task == 'multiclass':
            y_pred = np.clip(y_pred, eps, 1 - eps)
            if y_true.ndim == 1:
                # Convert to one-hot
                y_onehot = np.eye(y_pred.shape[1])[y_true]
            else:
                y_onehot = y_true
            return -np.mean(np.sum(y_onehot * np.log(y_pred), axis=1))

        elif task == 'regression':
            return np.mean((y_pred - y_true.reshape(-1, 1)) ** 2)

        return 0.0

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, task: str):
        """Backpropagation."""
        n = y_true.shape[0]

        # Gradient of loss w.r.t output layer
        if task == 'binary':
            delta = y_pred - y_true.reshape(-1, 1)
        elif task == 'multiclass':
            if y_true.ndim == 1:
                y_onehot = np.eye(y_pred.shape[1])[y_true]
            else:
                y_onehot = y_true
            delta = (y_pred - y_onehot)
        elif task == 'regression':
            delta = 2 * (y_pred - y_true.reshape(-1, 1))

        # Backpropagate through layers
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update_weights(self, lr: float, lambda_reg: float = 0.0):
        """Update weights with optional L2 regularization."""
        for layer in self.layers:
            layer.weights -= lr * (layer.dW + lambda_reg * layer.weights)
            layer.biases -= lr * layer.db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        lr: float = 0.01,
        batch_size: int = 32,
        task: str = 'binary',
        lambda_reg: float = 0.0,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: int = 100,
    ):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            epoch_loss = 0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_pred, y_batch, task)
                self.backward(y_pred, y_batch, task)
                self.update_weights(lr, lambda_reg)

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)

            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(val_pred, y_val, task)
                self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % verbose == 0:
                val_str = f", Val Loss: {self.val_losses[-1]:.4f}" if X_val is not None else ""
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}{val_str}")

    def predict(self, X: np.ndarray, task: str = 'binary') -> np.ndarray:
        y_pred = self.forward(X)
        if task == 'binary':
            return (y_pred >= 0.5).astype(int).flatten()
        elif task == 'multiclass':
            return np.argmax(y_pred, axis=1)
        return y_pred.flatten()

    def score(self, X: np.ndarray, y: np.ndarray, task: str = 'binary') -> float:
        y_pred = self.predict(X, task)
        if task in ['binary', 'multiclass']:
            return np.mean(y_pred == y)
        else:  # R² for regression
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            return 1 - ss_res / ss_tot

    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
```

## Backpropagation - Chi tiết toán học

```
Forward Pass:
  z[l] = W[l] · a[l-1] + b[l]
  a[l] = activation(z[l])

Loss:
  L = CrossEntropy(a[L], y)  # For classification

Backward Pass (Chain Rule):
  dL/da[L] = -(y/a[L]) + (1-y)/(1-a[L])  # dCrossEntropy/da

  dL/dz[L] = dL/da[L] · da[L]/dz[L]
            = dL/da[L] · activation'(z[L])

  dL/dW[l] = (1/m) · a[l-1]ᵀ · δ[l]
  dL/db[l] = (1/m) · Σ δ[l]

  δ[l-1] = (W[l])ᵀ · δ[l] · activation'(z[l-1])  # Backprop error
```

## Test Neural Network

```python
from sklearn.datasets import make_classification, make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Binary Classification - Moon dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Create network: 2 -> 16 -> 8 -> 1
nn = NeuralNetwork(
    layer_sizes=[2, 16, 8, 1],
    activations=['relu', 'relu', 'sigmoid']
)

nn.fit(
    X_train_s, y_train,
    epochs=500, lr=0.01, batch_size=32,
    task='binary',
    X_val=X_test_s, y_val=y_test,
    verbose=100,
)

print(f"\nTest Accuracy: {nn.score(X_test_s, y_test, 'binary'):.4f}")
nn.plot_training_history()

# Visualize decision boundary
h = 0.02
x_min, x_max = X_test_s[:, 0].min() - 1, X_test_s[:, 0].max() + 1
y_min, y_max = X_test_s[:, 1].min() - 1, X_test_s[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])[:, 0]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
plt.scatter(X_test_s[:, 0], X_test_s[:, 1], c=y_test, cmap='RdBu', edgecolors='k', s=40)
plt.title('Neural Network Decision Boundary')
plt.colorbar()
plt.show()
```

## Regularization Techniques

```python
# L2 Regularization (Weight Decay)
nn_l2 = NeuralNetwork([2, 32, 16, 1], ['relu', 'relu', 'sigmoid'])
nn_l2.fit(X_train_s, y_train, epochs=500, lr=0.01,
          task='binary', lambda_reg=0.01)  # lambda_reg = weight decay

# Dropout (implement separately)
class DropoutLayer:
    """Dropout regularization layer."""

    def __init__(self, keep_prob: float = 0.8):
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.mask = (np.random.rand(*X.shape) < self.keep_prob) / self.keep_prob
            return X * self.mask
        return X

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return delta * self.mask
```

---PRACTICE---

## Bài tập thực hành

### Bài tập 1: XOR Problem

Perceptron đơn không thể giải XOR. Chứng minh bằng code:
- Train Perceptron trên XOR gate
- Quan sát kết quả (sẽ fail)
- Build neural network 2-layer giải XOR
- Giải thích tại sao XOR cần ít nhất 1 hidden layer

### Bài tập 2: Neural Network cho MNIST

Sử dụng `NeuralNetwork` class đã implement:
- Load MNIST dataset từ sklearn
- Chuẩn bị data: flatten, normalize, one-hot encode labels
- Train network: [784, 256, 128, 10] với activations phù hợp
- Track training/validation accuracy
- Visualize: confusion matrix, sample predictions, learning curves
- Target: đạt ít nhất 95% test accuracy

### Bài tập 3: Gradient Checking

Implement gradient checking để verify backpropagation:
- Tính numerical gradient: `(f(θ+ε) - f(θ-ε)) / (2ε)`
- So sánh với analytical gradient từ backprop
- Sự khác biệt phải nhỏ hơn `1e-5`
- Test trên một mini neural network

---SOLUTION---

## Lời giải

### Giải bài tập 1: XOR Problem

```python
# xor_problem.py
import numpy as np
import matplotlib.pyplot as plt

# XOR truth table
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Perceptron CANNOT solve XOR
from neural_network import Perceptron  # Hoặc copy class Perceptron ở trên
p = Perceptron(n_features=2, learning_rate=0.1)
history = p.fit(X_xor, y_xor, epochs=100)
print("Perceptron predictions:", p.predict(X_xor))
print("Actual:                ", y_xor)
# Output sẽ sai vì XOR không linearly separable

# Neural Network có thể giải XOR!
from neural_network import NeuralNetwork

nn = NeuralNetwork(
    layer_sizes=[2, 4, 1],
    activations=['sigmoid', 'sigmoid']
)

nn.fit(
    X_xor, y_xor,
    epochs=5000,
    lr=0.5,
    batch_size=4,  # Cả dataset mỗi epoch (batch GD)
    task='binary',
    verbose=1000,
)

print("\nNeural Network predictions:")
raw_preds = nn.forward(X_xor)
binary_preds = nn.predict(X_xor, task='binary')
for i, (x, pred, actual) in enumerate(zip(X_xor, binary_preds, y_xor)):
    print(f"  Input: {x} -> Pred: {pred} (raw: {raw_preds[i,0]:.4f}), Actual: {actual}")

print(f"\nAccuracy: {nn.score(X_xor, y_xor, 'binary'):.0%}")

# Visualize: tại sao 1 hidden layer giải được XOR
# Hidden layer tạo ra không gian mới nơi XOR trở thành linearly separable
hidden_output = nn.layers[0].forward(X_xor)
print("\nHidden layer output (transformed space):")
for i, (x, h, y_val) in enumerate(zip(X_xor, hidden_output, y_xor)):
    print(f"  Input {x} -> Hidden {h.round(4)} -> Label {y_val}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original space (XOR is NOT linearly separable)
colors = ['red' if yi == 0 else 'blue' for yi in y_xor]
axes[0].scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200, zorder=5)
for i, (x, y_val) in enumerate(zip(X_xor, y_xor)):
    axes[0].annotate(f'({x[0]},{x[1]})={y_val}', xy=x,
                     xytext=(10, 10), textcoords='offset points')
axes[0].set_title('XOR in Original Space\n(NOT linearly separable)')
axes[0].set_xlim(-0.5, 1.5)
axes[0].set_ylim(-0.5, 1.5)

# Transformed space (linearly separable after hidden layer)
axes[1].scatter(hidden_output[:, 0], hidden_output[:, 1], c=colors, s=200, zorder=5)
for i, (h, y_val) in enumerate(zip(hidden_output, y_xor)):
    axes[1].annotate(f'={y_val}', xy=h[:2], xytext=(10, 10), textcoords='offset points')
axes[1].set_title('XOR in Hidden Layer Space\n(Linearly separable!)')

plt.tight_layout()
plt.show()
```

### Giải bài tập 2: MNIST với Neural Network từ đầu

```python
# mnist_nn.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits  # MNIST-like, dễ hơn full MNIST
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Load data
digits = load_digits()
X = digits.data        # (1797, 64) - 8x8 images flattened
y = digits.target      # (1797,) - labels 0-9

print(f"Dataset: {X.shape}, Labels: {np.unique(y)}")

# Normalize
X = X / 16.0  # Scale to [0, 1] (pixel values are 0-16)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Build network
nn = NeuralNetwork(
    layer_sizes=[64, 128, 64, 10],
    activations=['relu', 'relu', 'softmax']
)

print("Training...")
nn.fit(
    X_train, y_train,
    epochs=200,
    lr=0.05,
    batch_size=32,
    task='multiclass',
    X_val=X_val,
    y_val=y_val,
    verbose=50,
)

# Evaluate
test_acc = nn.score(X_test, y_test, 'multiclass')
print(f"\nTest Accuracy: {test_acc:.4f}")

# Confusion matrix
y_pred_test = nn.predict(X_test, 'multiclass')
cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.title('Confusion Matrix - Digit Recognition')
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10))
plt.yticks(tick_marks, range(10))
plt.xlabel('Predicted')
plt.ylabel('Actual')
for i in range(10):
    for j in range(10):
        plt.text(j, i, str(cm[i, j]),
                 ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.show()

# Visualize sample predictions
fig, axes = plt.subplots(2, 10, figsize=(20, 4))
for i in range(10):
    idx = np.where(y_test == i)[0][0]
    axes[0, i].imshow(X_test[idx].reshape(8, 8), cmap='gray')
    pred = y_pred_test[idx]
    correct = '✓' if pred == i else '✗'
    axes[0, i].set_title(f'True:{i}\nPred:{pred} {correct}', fontsize=8)
    axes[0, i].axis('off')

    # Wrong prediction
    wrong_idx = np.where((y_test == i) & (y_pred_test != i))
    if len(wrong_idx[0]) > 0:
        widx = wrong_idx[0][0]
        axes[1, i].imshow(X_test[widx].reshape(8, 8), cmap='gray')
        axes[1, i].set_title(f'True:{i}\nPred:{y_pred_test[widx]} ✗', fontsize=8, color='red')
    else:
        axes[1, i].set_title('No errors!', fontsize=8, color='green')
    axes[1, i].axis('off')

plt.suptitle('Sample Predictions (Row 1: Correct, Row 2: Wrong)')
plt.tight_layout()
plt.show()
```

### Giải bài tập 3: Gradient Checking

```python
# gradient_checking.py
import numpy as np

def compute_numerical_gradient(nn, X, y, task, eps=1e-5):
    """
    Tính numerical gradient sử dụng finite difference.
    Dùng để verify backpropagation.
    """
    numerical_grads = []

    for layer_idx, layer in enumerate(nn.layers):
        # Check weights
        W = layer.weights
        dW_num = np.zeros_like(W)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                # f(θ + ε)
                W[i, j] += eps
                y_pred_plus = nn.forward(X)
                loss_plus = nn.compute_loss(y_pred_plus, y, task)

                # f(θ - ε)
                W[i, j] -= 2 * eps
                y_pred_minus = nn.forward(X)
                loss_minus = nn.compute_loss(y_pred_minus, y, task)

                # Restore
                W[i, j] += eps

                # Numerical gradient
                dW_num[i, j] = (loss_plus - loss_minus) / (2 * eps)

        numerical_grads.append(dW_num)

    return numerical_grads

def gradient_check(nn, X, y, task='binary', eps=1e-5, threshold=1e-5):
    """
    So sánh analytical gradient (backprop) với numerical gradient.
    """
    # 1. Forward pass
    y_pred = nn.forward(X)

    # 2. Backward pass (analytical gradients)
    nn.backward(y_pred, y, task)
    analytical_grads = [layer.dW.copy() for layer in nn.layers]

    # 3. Numerical gradients
    numerical_grads = compute_numerical_gradient(nn, X, y, task, eps)

    # 4. Compare
    all_passed = True
    for i, (analyt, numer) in enumerate(zip(analytical_grads, numerical_grads)):
        diff = np.abs(analyt - numer)
        max_diff = diff.max()
        rel_error = np.linalg.norm(analyt - numer) / (np.linalg.norm(analyt) + np.linalg.norm(numer) + 1e-15)

        status = "PASS" if rel_error < threshold else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"Layer {i} - Max diff: {max_diff:.2e}, Relative error: {rel_error:.2e} [{status}]")

    return all_passed

# Test gradient checking
np.random.seed(42)
X_small = np.random.randn(5, 4)  # Small dataset cho speed
y_small = np.random.randint(0, 2, 5)  # Binary labels

nn_check = NeuralNetwork(
    layer_sizes=[4, 8, 1],
    activations=['tanh', 'sigmoid']  # Tanh để gradient không bị clip
)

print("Gradient Checking (Binary Classification):")
passed = gradient_check(nn_check, X_small, y_small, task='binary')
print(f"\nAll gradients correct: {passed}")
```
