# Python cho Machine Learning

## Tại sao Python cho ML?

Python chiếm hơn **85%** codebase trong lĩnh vực ML/AI vì:
- **Cú pháp đơn giản**: Tập trung vào logic, không phải syntax
- **Ecosystem phong phú**: NumPy, Pandas, Scikit-learn, PyTorch, TensorFlow
- **Cộng đồng lớn**: Nhiều tài liệu, tutorials, Stack Overflow answers
- **Tính tương tác**: Jupyter Notebook cho phép thử nghiệm nhanh
- **Glue language**: Dễ dàng kết nối với C/C++/CUDA libraries

## Cài đặt môi trường ML

```bash
# Cài Anaconda (khuyến nghị cho ML)
# Download từ: https://www.anaconda.com/products/distribution

# Tạo environment riêng cho ML
conda create -n ml_env python=3.11
conda activate ml_env

# Cài đặt thư viện cơ bản
pip install numpy pandas matplotlib scikit-learn jupyter
pip install torch torchvision  # PyTorch

# Hoặc dùng venv
python -m venv ml_env
source ml_env/bin/activate  # Linux/Mac
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Python Essentials cho ML

### List Comprehensions

```python
# Cách thông thường
squares = []
for x in range(10):
    squares.append(x ** 2)

# List comprehension - Pythonic hơn và NHANH hơn (~35%)
squares = [x ** 2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Với điều kiện
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
# [0, 4, 16, 36, 64]

# Nested list comprehension - Tạo matrix
matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]
# [[1, 2, 3], [2, 4, 6], [3, 6, 9]]

# Dict comprehension
word_lengths = {word: len(word) for word in ["python", "machine", "learning"]}
# {'python': 6, 'machine': 7, 'learning': 8}

# Set comprehension
unique_lengths = {len(word) for word in ["ai", "ml", "nlp", "cv"]}
# {2, 3}
```

### Generators - Xử lý data lớn

```python
# Generator expressions - Không load hết vào RAM
# Dùng khi data lớn hoặc cần lazy evaluation
gen = (x ** 2 for x in range(1_000_000))  # Không tạo list ngay

# Chỉ tính khi cần
print(next(gen))   # 0
print(next(gen))   # 1
print(sum(gen))    # Tính tổng (tốn ít RAM hơn list)

# Generator function cho data pipeline
def batch_generator(data: list, batch_size: int):
    """
    Yield data theo từng batch.
    Dùng trong ML khi data quá lớn để load toàn bộ.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Sử dụng
data = list(range(1000))
for batch in batch_generator(data, batch_size=32):
    # Process từng batch (như trong training loop)
    pass

# Real-world: generator cho CSV lớn
def read_large_csv(file_path: str, batch_size: int = 1000):
    """Đọc CSV lớn theo batch."""
    import csv
    with open(file_path) as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
```

### Decorators

```python
import time
import functools

# Decorator cơ bản - Timing function
def timer(func):
    """Đo thời gian chạy của function."""
    @functools.wraps(func)  # Giữ nguyên metadata của func gốc
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} chạy trong {end - start:.4f}s")
        return result
    return wrapper

@timer
def train_model(epochs: int):
    """Giả lập training."""
    time.sleep(epochs * 0.1)
    return "model trained"

train_model(5)  # Output: "train_model chạy trong 0.5002s"

# Decorator với tham số
def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry function nếu gặp lỗi."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def download_data(url: str):
    """Download data với retry."""
    import urllib.request
    urllib.request.urlretrieve(url, "data.csv")
    return "Downloaded!"

# Cache decorator - Memoization
def memoize(func):
    """Cache kết quả của function."""
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Python built-in: functools.lru_cache
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(n: int) -> float:
    """Tính toán tốn thời gian - được cache tự động."""
    return sum(i ** 2 for i in range(n))
```

### Object-Oriented Programming cho ML

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class BaseModel(ABC):
    """
    Abstract base class cho tất cả ML models.
    Định nghĩa interface chung.
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self._params = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Train model. Phải return self để cho phép method chaining."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán."""
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Tính accuracy. Có thể override trong subclass."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_params(self) -> dict:
        """Trả về hyperparameters."""
        return self._params.copy()

    def set_params(self, **params) -> 'BaseModel':
        """Set hyperparameters. Trả về self để method chaining."""
        self._params.update(params)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"

class SimpleLinearRegression(BaseModel):
    """Linear Regression từ đầu."""

    def __init__(self):
        super().__init__("Linear Regression")
        self.weights = None
        self.bias = None
        self._params = {"learning_rate": 0.01, "epochs": 1000}

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleLinearRegression':
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        lr = self._params["learning_rate"]
        epochs = self._params["epochs"]

        for epoch in range(epochs):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Gradients
            dw = (2 / n_samples) * X.T @ (y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= lr * dw
            self.bias -= lr * db

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model chưa được train. Gọi fit() trước.")
        return X @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Override: dùng R² score thay vì accuracy."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

# Sử dụng
model = SimpleLinearRegression()
model.set_params(learning_rate=0.001, epochs=2000)

# Tạo data
X = np.random.randn(100, 1)
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 0.1

model.fit(X, y)
r2 = model.score(X, y)
print(f"R² score: {r2:.4f}")
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias:.4f}")
print(model)  # __repr__
```

### Context Managers

```python
import time
from contextlib import contextmanager

@contextmanager
def timer_context(label: str = ""):
    """Context manager để đo thời gian."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s")

# Sử dụng
with timer_context("Matrix multiplication"):
    A = np.random.randn(1000, 1000)
    B = np.random.randn(1000, 1000)
    C = A @ B

# Experiment tracking context
@contextmanager
def experiment(name: str):
    """Track ML experiment."""
    print(f"Starting experiment: {name}")
    metadata = {"name": name, "start_time": time.time(), "metrics": {}}
    try:
        yield metadata
    finally:
        metadata["duration"] = time.time() - metadata["start_time"]
        print(f"Experiment '{name}' finished in {metadata['duration']:.2f}s")
        print(f"Metrics: {metadata['metrics']}")

with experiment("Linear Regression Test") as exp:
    # Train
    model = SimpleLinearRegression()
    model.fit(X, y)
    # Record metrics
    exp["metrics"]["r2"] = model.score(X, y)
    exp["metrics"]["weights"] = model.weights.tolist()
```

### Type Hints và Dataclasses

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

@dataclass
class TrainingConfig:
    """Configuration cho training ML model."""
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10
    model_name: str = "model"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])

    def __post_init__(self):
        """Validate sau khi khởi tạo."""
        assert 0 < self.learning_rate < 1, "Learning rate phải trong (0, 1)"
        assert self.epochs > 0, "Epochs phải > 0"
        assert 0 < self.validation_split < 1, "Validation split phải trong (0, 1)"

@dataclass
class TrainingResult:
    """Kết quả sau khi train."""
    train_loss: List[float]
    val_loss: List[float]
    best_epoch: int
    best_val_loss: float
    model_path: Optional[str] = None

    @property
    def is_overfit(self) -> bool:
        """Kiểm tra overfitting đơn giản."""
        if len(self.train_loss) < 2:
            return False
        return self.val_loss[-1] > self.train_loss[-1] * 1.5

# Sử dụng
config = TrainingConfig(
    learning_rate=0.01,
    epochs=50,
    batch_size=64,
    model_name="experiment_1"
)
print(config)
```

---PRACTICE---

## Bài tập thực hành

### Bài tập 1: Data Pipeline với Generators

Xây dựng một data pipeline hoàn chỉnh sử dụng generators:
- `load_csv_batches(file_path, batch_size)`: Load CSV file theo từng batch
- `normalize_batch(batch)`: Normalize numerical columns về [0, 1]
- `encode_labels(batch, label_col)`: Encode categorical labels thành integers
- `split_features_labels(batch, label_col)`: Tách features và labels
- Pipeline phải xử lý được file CSV 1GB+ mà không OOM

### Bài tập 2: Scikit-learn compatible Model

Xây dựng custom ML model tuân theo Scikit-learn API:
- Implement `fit(X, y)`, `predict(X)`, `predict_proba(X)`
- Sử dụng `check_estimator` từ sklearn để validate
- Model: K-Nearest Neighbors từ đầu (không dùng sklearn.neighbors)
- Support cả classification và regression (detect tự động từ y)

### Bài tập 3: Experiment Tracker

Xây dựng một experiment tracker đơn giản:
- `Experiment` dataclass với config, metrics, artifacts
- Lưu/load experiment từ JSON file
- So sánh nhiều experiments
- Plot learning curves với matplotlib

---SOLUTION---

## Lời giải

### Giải bài tập 1: Data Pipeline

```python
# data_pipeline.py
import csv
import numpy as np
from typing import Generator, List, Dict, Any

def load_csv_batches(
    file_path: str,
    batch_size: int = 1000
) -> Generator[List[Dict], None, None]:
    """Load CSV file theo từng batch."""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def normalize_batch(
    batch: List[Dict],
    numeric_cols: List[str],
    stats: Dict = None
) -> tuple:
    """
    Normalize numerical columns về [0, 1].
    Nếu stats=None, tính stats từ batch hiện tại (fit).
    Nếu stats được cung cấp, dùng stats đó (transform).
    """
    if stats is None:
        stats = {}
        for col in numeric_cols:
            values = [float(row[col]) for row in batch if row[col] != '']
            stats[col] = {"min": min(values), "max": max(values)}

    normalized = []
    for row in batch:
        new_row = row.copy()
        for col in numeric_cols:
            if row[col] != '':
                val = float(row[col])
                col_min = stats[col]["min"]
                col_max = stats[col]["max"]
                if col_max > col_min:
                    new_row[col] = (val - col_min) / (col_max - col_min)
                else:
                    new_row[col] = 0.0
        normalized.append(new_row)

    return normalized, stats

def encode_labels(
    batch: List[Dict],
    label_col: str,
    label_map: Dict = None
) -> tuple:
    """Encode categorical labels thành integers."""
    if label_map is None:
        unique_labels = sorted(set(row[label_col] for row in batch))
        label_map = {label: i for i, label in enumerate(unique_labels)}

    encoded = []
    for row in batch:
        new_row = row.copy()
        if row[label_col] in label_map:
            new_row[label_col] = label_map[row[label_col]]
        encoded.append(new_row)

    return encoded, label_map

def split_features_labels(
    batch: List[Dict],
    label_col: str,
    feature_cols: List[str]
) -> tuple:
    """Tách features và labels thành numpy arrays."""
    X = np.array([
        [float(row[col]) for col in feature_cols]
        for row in batch
    ])
    y = np.array([row[label_col] for row in batch])
    return X, y

class DataPipeline:
    """Pipeline hoàn chỉnh cho ML data processing."""

    def __init__(
        self,
        file_path: str,
        label_col: str,
        numeric_cols: List[str],
        batch_size: int = 1000
    ):
        self.file_path = file_path
        self.label_col = label_col
        self.numeric_cols = numeric_cols
        self.batch_size = batch_size
        self.stats = None
        self.label_map = None

    def fit_transform(self) -> Generator[tuple, None, None]:
        """Fit pipeline trên first batch, transform tất cả."""
        is_first_batch = True

        for batch in load_csv_batches(self.file_path, self.batch_size):
            # Normalize
            if is_first_batch:
                normalized, self.stats = normalize_batch(batch, self.numeric_cols)
                encoded, self.label_map = encode_labels(normalized, self.label_col)
                is_first_batch = False
            else:
                normalized, _ = normalize_batch(batch, self.numeric_cols, self.stats)
                encoded, _ = encode_labels(normalized, self.label_col, self.label_map)

            # Split
            feature_cols = [c for c in self.numeric_cols if c != self.label_col]
            X, y = split_features_labels(encoded, self.label_col, feature_cols)
            yield X, y
```

### Giải bài tập 2: KNN từ đầu

```python
# knn.py
import numpy as np
from collections import Counter

class KNearestNeighbors:
    """
    K-Nearest Neighbors classifier và regressor từ đầu.
    Tuân theo Scikit-learn API.
    """

    def __init__(self, k: int = 5, task: str = "auto"):
        self.k = k
        self.task = task  # "classification", "regression", hoặc "auto"
        self.X_train = None
        self.y_train = None
        self._task = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNearestNeighbors':
        self.X_train = X.copy()
        self.y_train = y.copy()

        # Auto-detect task
        if self.task == "auto":
            unique = np.unique(y)
            self._task = "classification" if len(unique) < 20 else "regression"
        else:
            self._task = self.task

        return self

    def _euclidean_distances(self, X: np.ndarray) -> np.ndarray:
        """Tính Euclidean distance từ X đến tất cả training points."""
        # Vectorized computation: ||x - x_train||²
        # = ||x||² + ||x_train||² - 2 * x · x_train
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)          # (n_test, 1)
        X_train_sq = np.sum(self.X_train ** 2, axis=1)         # (n_train,)
        cross = X @ self.X_train.T                              # (n_test, n_train)
        distances = np.sqrt(np.maximum(X_sq + X_train_sq - 2 * cross, 0))
        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            raise RuntimeError("Chưa được fit. Gọi fit() trước.")

        distances = self._euclidean_distances(X)
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]

        predictions = []
        for indices in k_nearest_indices:
            k_labels = self.y_train[indices]
            if self._task == "classification":
                # Majority vote
                prediction = Counter(k_labels).most_common(1)[0][0]
            else:
                # Mean for regression
                prediction = np.mean(k_labels)
            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Xác suất cho classification."""
        if self._task != "classification":
            raise ValueError("predict_proba chỉ dùng cho classification.")

        distances = self._euclidean_distances(X)
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        classes = np.unique(self.y_train)

        probabilities = []
        for indices in k_nearest_indices:
            k_labels = self.y_train[indices]
            proba = [np.mean(k_labels == c) for c in classes]
            probabilities.append(proba)

        return np.array(probabilities)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        if self._task == "classification":
            return np.mean(y_pred == y)
        else:  # regression: R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot

# Test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNearestNeighbors(k=5, task="classification")
knn.fit(X_train, y_train)
print(f"KNN Accuracy: {knn.score(X_test, y_test):.4f}")

# So sánh với sklearn
from sklearn.neighbors import KNeighborsClassifier
sklearn_knn = KNeighborsClassifier(n_neighbors=5)
sklearn_knn.fit(X_train, y_train)
print(f"Sklearn KNN Accuracy: {sklearn_knn.score(X_test, y_test):.4f}")
```

### Giải bài tập 3: Experiment Tracker

```python
# experiment_tracker.py
import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import matplotlib.pyplot as plt

@dataclass
class ExperimentConfig:
    model_name: str
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Experiment:
    name: str
    config: ExperimentConfig
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    notes: str = ""

    def log_epoch(self, train_loss: float, val_loss: float,
                  train_acc: float = None, val_acc: float = None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)

    def finish(self, **metrics):
        self.end_time = time.time()
        self.final_metrics = metrics

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def best_val_loss(self) -> float:
        return min(self.val_losses) if self.val_losses else float('inf')

    def save(self, path: str):
        data = asdict(self)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Experiment':
        with open(path) as f:
            data = json.load(f)
        config = ExperimentConfig(**data.pop('config'))
        return cls(config=config, **data)

    def plot(self, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_title(f'Loss - {self.name}')
        axes[0].legend()
        axes[0].grid(True)

        if self.train_accuracies:
            axes[1].plot(self.train_accuracies, label='Train Acc')
            axes[1].plot(self.val_accuracies, label='Val Acc')
            axes[1].set_title('Accuracy')
            axes[1].legend()
            axes[1].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

# Demo
exp = Experiment(
    name="experiment_1",
    config=ExperimentConfig(model_name="LinearRegression", learning_rate=0.01)
)

for epoch in range(50):
    train_loss = 1.0 - epoch * 0.015 + np.random.normal(0, 0.02)
    val_loss = 1.0 - epoch * 0.012 + np.random.normal(0, 0.03)
    exp.log_epoch(max(0.01, train_loss), max(0.01, val_loss))

exp.finish(final_val_loss=exp.best_val_loss, epochs_trained=50)
exp.save("experiment_1.json")
print(f"Best val loss: {exp.best_val_loss:.4f}")
print(f"Duration: {exp.duration:.2f}s")
```
