# Deep Learning với PyTorch

## PyTorch là gì?

**PyTorch** là framework deep learning mã nguồn mở do Meta AI phát triển, nổi tiếng với:
- **Dynamic computation graph**: Dễ debug, linh hoạt hơn TensorFlow (static graph)
- **Pythonic**: Cảm giác như viết Python thuần
- **Research-friendly**: Được ưa chuộng trong academia
- **Production-ready**: TorchServe, TorchScript, ONNX export

## Cài đặt

```bash
# CPU only
pip install torch torchvision

# Với GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Kiểm tra
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## PyTorch Tensors

```python
import torch
import numpy as np

# Tạo tensors
t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.zeros(3, 4)
t3 = torch.ones(2, 3, 4)  # 3D tensor
t4 = torch.randn(5, 5)     # Standard normal
t5 = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
t6 = torch.linspace(0, 1, 100)

# Data types
t_float = torch.tensor([1.0, 2.0], dtype=torch.float32)  # Default cho neural nets
t_int = torch.tensor([1, 2, 3], dtype=torch.int64)       # Default cho labels
t_bool = torch.tensor([True, False, True])

# Numpy <-> Tensor (ZERO-COPY nếu cùng device)
np_array = np.array([1.0, 2.0, 3.0])
tensor_from_np = torch.from_numpy(np_array)  # Shares memory!
np_from_tensor = t1.numpy()                  # Shares memory!

# Di chuyển giữa CPU và GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

t_cpu = torch.randn(3, 4)
t_gpu = t_cpu.to(device)  # Chuyển lên GPU

# Tensor operations
a = torch.randn(3, 4)
b = torch.randn(4, 5)

print(a.shape)         # torch.Size([3, 4])
print(a + 1)           # Element-wise
print(a * 2)           # Element-wise
print(a @ b)           # Matrix multiplication (3, 5)
print(torch.matmul(a, b))  # Same
print(a.T)             # Transpose (4, 3)
print(a.sum())         # Sum all
print(a.mean(dim=0))   # Mean along dim 0
print(a.max())         # Global max
print(a.argmax())      # Index of global max
print(a.reshape(2, 6)) # Reshape (must same size)
print(a.view(2, 6))    # Like reshape but shared memory
print(a.unsqueeze(0))  # Add dim: (1, 3, 4)
print(a.squeeze())     # Remove dims of size 1
```

## Autograd - Automatic Differentiation

```python
import torch

# requires_grad=True: tính gradient cho tensor này
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward pass: tạo computation graph tự động
z = x ** 2 + 2 * x * y + y ** 2  # z = (x+y)² = 25

# Backward pass: tính gradients
z.backward()

# dz/dx = 2x + 2y = 2(2) + 2(3) = 10
print(f"dz/dx = {x.grad}")  # tensor(10.)
# dz/dy = 2x + 2y = 10
print(f"dz/dy = {y.grad}")  # tensor(10.)

# Gradient với non-scalar output
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = x ** 2
y.backward(torch.ones_like(y))  # Gradient of sum(y²) w.r.t x
print(f"dx = {x.grad}")  # [[2, 4], [6, 8]] = 2x

# Tắt gradient tracking (inference, tăng tốc)
with torch.no_grad():
    result = x ** 2  # Không tạo computation graph

# Detach tensor khỏi computation graph
detached = x.detach()
```

## nn.Module - Building Blocks

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Layer
class CustomLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # nn.Parameter: được track và update bởi optimizer
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias

# Fully Connected Network
class FCNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 dropout: float = 0.3):
        super().__init__()

        # Xây dựng layers động
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Kiểm tra model
model = FCNetwork(784, [256, 128, 64], 10)
print(model)

# Đếm parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test forward pass
x = torch.randn(32, 784)  # Batch of 32 samples
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 10)
```

## Loss Functions và Optimizers

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Loss Functions
# Classification
ce_loss = nn.CrossEntropyLoss()          # Multi-class (applies softmax internally)
bce_loss = nn.BCEWithLogitsLoss()        # Binary (applies sigmoid internally)
nll_loss = nn.NLLLoss()                  # Với log_softmax output

# Regression
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
huber_loss = nn.HuberLoss(delta=1.0)     # Kết hợp MSE và MAE

# Optimizers
model = FCNetwork(10, [32], 5)

sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Better regularization
rmsprop = optim.RMSprop(model.parameters(), lr=0.001)

# Learning Rate Schedulers
scheduler_step = optim.lr_scheduler.StepLR(adam, step_size=10, gamma=0.5)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(adam, T_max=100)
scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(adam, patience=5, factor=0.5)
scheduler_warmup = optim.lr_scheduler.LinearLR(adam, start_factor=0.1, total_iters=10)
```

## Training Loop hoàn chỉnh

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

class Trainer:
    """
    Training loop chuẩn cho PyTorch models.
    Hỗ trợ: early stopping, LR scheduling, checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'auto',
        scheduler=None,
    ):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        ) if device == 'auto' else torch.device(device)

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self, dataloader: DataLoader) -> tuple:
        self.model.train()  # QUAN TRỌNG: bật training mode (dropout, batchnorm)
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Zero gradients (tích lũy từ step trước)
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(X_batch)
            loss = self.criterion(output, y_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping (ngăn gradient explosion)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Tracking
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += X_batch.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()  # Tắt gradient computation cho evaluation
    def eval_epoch(self, dataloader: DataLoader) -> tuple:
        self.model.eval()  # QUAN TRỌNG: tắt training mode
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            output = self.model(X_batch)
            loss = self.criterion(output, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += X_batch.size(0)

        return total_loss / total, correct / total

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        save_best: bool = True,
    ):
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.eval_epoch(val_loader)

            # Tracking
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping + model checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_best:
                    self.best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                patience_counter = 0
            else:
                patience_counter += 1

            current_lr = self.optimizer.param_groups[0]['lr']
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.6f}")

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if save_best and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"Restored best model (val_loss={self.best_val_loss:.4f})")

    def plot_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_title('Loss History')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.train_accs, label='Train Accuracy')
        axes[1].plot(self.val_accs, label='Val Accuracy')
        axes[1].set_title('Accuracy History')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Checkpoint loaded from {path}")

# Demo: Training on digits dataset
digits = load_digits()
X = digits.data.astype(np.float32)
y = digits.target

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

# Create tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y).long()

dataset = TensorDataset(X_tensor, y_tensor)
n_train = int(0.7 * len(dataset))
n_val = int(0.15 * len(dataset))
n_test = len(dataset) - n_train - n_val

train_set, val_set, test_set = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Model
model = FCNetwork(64, [256, 128], 10, dropout=0.3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Train
trainer = Trainer(model, criterion, optimizer, scheduler=scheduler)
trainer.fit(train_loader, val_loader, epochs=100, early_stopping_patience=15)

# Test
test_loss, test_acc = trainer.eval_epoch(test_loader)
print(f"\nTest Accuracy: {test_acc:.4f}")
trainer.plot_history()
```

## CNN - Convolutional Neural Networks

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Chuẩn hóa Conv-BN-ReLU block."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class SimpleCNN(nn.Module):
    """
    CNN cho image classification.
    Input: (batch, channels, height, width)
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # (B, 32, H, W)
            ConvBlock(32, 32),             # (B, 32, H, W)
            nn.MaxPool2d(2, 2),            # (B, 32, H/2, W/2)
            nn.Dropout2d(0.25),

            ConvBlock(32, 64),             # (B, 64, H/2, W/2)
            ConvBlock(64, 64),             # (B, 64, H/2, W/2)
            nn.MaxPool2d(2, 2),            # (B, 64, H/4, W/4)
            nn.Dropout2d(0.25),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

# Test CNN
cnn = SimpleCNN(num_classes=10, in_channels=1)
x = torch.randn(4, 1, 28, 28)  # 4 MNIST-like images
output = cnn(x)
print(f"CNN output shape: {output.shape}")  # (4, 10)

total_params = sum(p.numel() for p in cnn.parameters())
print(f"CNN parameters: {total_params:,}")
```

## Saving và Loading Models

```python
import torch

# Lưu toàn bộ model (không khuyến nghị, phụ thuộc vào class definition)
torch.save(model, 'model_full.pt')
loaded_model = torch.load('model_full.pt')

# Chỉ lưu state dict (KHUYẾN NGHỊ)
torch.save(model.state_dict(), 'model_weights.pt')

# Load lại
new_model = FCNetwork(64, [256, 128], 10)  # Phải tạo lại model cùng architecture
new_model.load_state_dict(torch.load('model_weights.pt'))
new_model.eval()  # Bắt buộc cho inference

# Checkpoint đầy đủ để resume training
checkpoint = {
    'epoch': 50,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'best_val_loss': 0.15,
    'train_losses': trainer.train_losses,
    'val_losses': trainer.val_losses,
    'config': {
        'hidden_dims': [256, 128],
        'dropout': 0.3,
        'learning_rate': 0.001,
    }
}
torch.save(checkpoint, 'checkpoint_epoch50.pt')

# Load và resume
checkpoint = torch.load('checkpoint_epoch50.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
print(f"Resuming from epoch {start_epoch}")

# ONNX export (cross-platform deployment)
dummy_input = torch.randn(1, 64)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
)
```

---PRACTICE---

## Bài tập thực hành

### Bài tập 1: Custom Dataset và DataLoader

Xây dựng một custom Dataset class cho text classification:
- Data: list of (text, label) tuples
- Tokenize text (word-level)
- Pad sequences đến độ dài cố định
- Implement `__len__` và `__getitem__`
- Tạo DataLoader với custom collate_fn
- Train một simple model trên IMDB-like data

### Bài tập 2: Transfer Learning

Sử dụng pre-trained model để classify custom dataset:
- Load ResNet18 pre-trained từ torchvision
- Freeze các layers đầu (feature extractor)
- Thay thế final layer cho task của bạn (e.g., 5 classes)
- Fine-tune chỉ final layer (linear probe)
- Sau đó fine-tune toàn bộ (full fine-tuning)
- So sánh kết quả và training time

### Bài tập 3: Implement Simple Transformer

Implement Transformer encoder block từ đầu:
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Positional encoding
- Stack 2 blocks và dùng cho sequence classification

---SOLUTION---

## Lời giải

### Giải bài tập 1: Custom Dataset

```python
# custom_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter

class TextDataset(Dataset):
    """Custom Dataset cho text classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: dict = None,
        max_len: int = 100,
    ):
        self.labels = labels
        self.max_len = max_len

        # Build vocabulary nếu chưa có
        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab

        # Tokenize và encode
        self.encoded = [self._encode(text) for text in texts]

    def _build_vocab(self, texts: list[str], min_freq: int = 2) -> dict:
        """Xây dựng vocabulary từ texts."""
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())

        # Chỉ giữ từ xuất hiện >= min_freq lần
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, freq in counter.items():
            if freq >= min_freq:
                vocab[word] = len(vocab)

        return vocab

    def _encode(self, text: str) -> list[int]:
        """Chuyển text thành list of token IDs."""
        tokens = text.lower().split()[:self.max_len]
        return [self.vocab.get(token, 1) for token in tokens]  # 1 = <UNK>

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        return self.encoded[idx], self.labels[idx]

def collate_fn(batch: list) -> tuple:
    """
    Pad sequences trong batch đến cùng độ dài.
    Called automatically bởi DataLoader.
    """
    sequences, labels = zip(*batch)

    # Tìm độ dài dài nhất trong batch
    max_len = max(len(seq) for seq in sequences)

    # Pad với 0 (<PAD>)
    padded = []
    lengths = []
    for seq in sequences:
        lengths.append(len(seq))
        pad_len = max_len - len(seq)
        padded.append(seq + [0] * pad_len)

    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),  # Useful cho LSTM
    )

# Simple Text Classifier với Embedding
class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) -> embedded: (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        # Mean pooling: (batch, embed_dim)
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)

# Generate sample text data
np.random.seed(42)
positive_templates = [
    "this movie is great excellent wonderful",
    "loved every moment of this film amazing",
    "fantastic performance best movie ever",
    "highly recommend this outstanding film",
]
negative_templates = [
    "terrible boring waste of time awful",
    "worst movie ever seen horrible acting",
    "completely disappointing bad production",
    "avoid this trash terrible writing",
]

texts = []
labels = []
for _ in range(200):
    if np.random.random() > 0.5:
        texts.append(np.random.choice(positive_templates))
        labels.append(1)
    else:
        texts.append(np.random.choice(negative_templates))
        labels.append(0)

# Create datasets
split = int(0.8 * len(texts))
train_dataset = TextDataset(texts[:split], labels[:split], max_len=50)
test_dataset = TextDataset(texts[split:], labels[split:],
                            vocab=train_dataset.vocab, max_len=50)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Train
model = TextClassifier(
    vocab_size=len(train_dataset.vocab),
    embed_dim=32, hidden_dim=64, num_classes=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    for X_batch, y_batch, lengths in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

print("Text classifier trained!")
```

### Giải bài tập 2: Transfer Learning

```python
# transfer_learning.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

def create_transfer_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Load ResNet18 và modify cho custom classification task.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze tất cả layers
    for param in model.parameters():
        param.requires_grad = False

    # Thay thế final FC layer (không frozen)
    in_features = model.fc.in_features  # 512 cho ResNet18
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    # fc layer có requires_grad=True by default

    return model

def unfreeze_layers(model, layers_to_unfreeze: int = 1):
    """
    Unfreeze các layers cuối để full fine-tuning.
    """
    # Lấy tất cả children modules
    children = list(model.children())

    # Unfreeze từ cuối
    for child in children[-layers_to_unfreeze:]:
        for param in child.parameters():
            param.requires_grad = True

    return model

# Demo (với random data vì không có real dataset)
num_classes = 5
batch_size = 16

# Fake data: ImageNet-like (3 channels, 224x224)
X_fake = torch.randn(100, 3, 224, 224)
y_fake = torch.randint(0, num_classes, (100,))

dataset = TensorDataset(X_fake, y_fake)
train_size = 80
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, 20])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Phase 1: Linear Probe (only train final layer)
print("Phase 1: Linear Probe")
model = create_transfer_model(num_classes).to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Quick training
model.train()
for batch_X, batch_y in train_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    optimizer.zero_grad()
    loss = criterion(model(batch_X), batch_y)
    loss.backward()
    optimizer.step()

# Phase 2: Full Fine-tuning
print("\nPhase 2: Full Fine-tuning")
model = unfreeze_layers(model, layers_to_unfreeze=3)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for fine-tuning!
```

### Giải bài tập 3: Simple Transformer

```python
# simple_transformer.py
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention từ đầu."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape  # Batch, Sequence length, Channels

        # Project and split into Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        Q, K, V = qkv.unbind(0)

        # Scaled dot-product attention
        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale  # (B, heads, T, T)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        out = attn_weights @ V  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)

        return self.out_proj(out)

class TransformerBlock(nn.Module):
    """Transformer encoder block: Attention + FFN + LayerNorm."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm (modern transformers use this)
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

        self.register_buffer('pe', pe)  # Không phải parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerClassifier(nn.Module):
    """Transformer cho sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        max_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(embedded)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # CLS token: lấy representation của token đầu tiên
        cls_repr = x[:, 0]
        return self.classifier(cls_repr)

# Test
vocab_size = 1000
model = TransformerClassifier(vocab_size=vocab_size, num_classes=2)

x = torch.randint(0, vocab_size, (8, 50))  # 8 sequences of length 50
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")  # (8, 2)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")
```
