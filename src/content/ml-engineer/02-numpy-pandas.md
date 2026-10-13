# NumPy & Pandas

## NumPy - Nền tảng của tính toán số trong Python

**NumPy** (Numerical Python) cung cấp đối tượng `ndarray` - mảng N chiều lưu trữ các phần tử cùng kiểu, cực kỳ nhanh nhờ tối ưu hóa bằng C và BLAS/LAPACK.

### Tại sao NumPy nhanh hơn Python list?

```python
import numpy as np
import time

size = 1_000_000

# Python list: loop trong Python (chậm)
start = time.perf_counter()
python_list = list(range(size))
result_list = [x ** 2 for x in python_list]
print(f"Python list: {time.perf_counter() - start:.3f}s")

# NumPy: vectorized operations trong C (rất nhanh)
start = time.perf_counter()
arr = np.arange(size)
result_arr = arr ** 2
print(f"NumPy: {time.perf_counter() - start:.3f}s")
# NumPy thường nhanh hơn 10-100x
```

## Tạo Arrays

```python
import numpy as np

# Từ Python list
arr1d = np.array([1, 2, 3, 4, 5])               # 1D: shape (5,)
arr2d = np.array([[1, 2, 3], [4, 5, 6]])         # 2D: shape (2, 3)
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D: shape (2, 2, 2)

# Các hàm tạo array đặc biệt
zeros = np.zeros((3, 4))           # Ma trận 3x4 toàn 0
ones = np.ones((2, 3), dtype=int)  # Ma trận 2x3 toàn 1
eye = np.eye(4)                    # Ma trận đơn vị 4x4
full = np.full((3, 3), 7.0)       # Ma trận 3x3 toàn giá trị 7

# Sequences
linspace = np.linspace(0, 1, 100)        # 100 điểm đều từ 0 đến 1
arange = np.arange(0, 10, 0.5)          # Từ 0 đến 10, bước 0.5
logspace = np.logspace(0, 3, 4)         # [1, 10, 100, 1000]

# Random
np.random.seed(42)  # Để kết quả tái lập
rand_uniform = np.random.rand(3, 3)          # Uniform [0, 1)
rand_normal = np.random.randn(3, 3)          # Standard normal N(0, 1)
rand_int = np.random.randint(0, 10, (3, 3)) # Integers [0, 10)
rand_choice = np.random.choice([1, 2, 3, 4, 5], size=10, replace=False)
```

## Array Properties và Indexing

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"Shape: {arr.shape}")        # (3, 3)
print(f"Ndim: {arr.ndim}")          # 2
print(f"Size: {arr.size}")          # 9
print(f"Dtype: {arr.dtype}")        # int64
print(f"Itemsize: {arr.itemsize}")  # 8 bytes

# Basic indexing
print(arr[0, 1])    # 2 (row 0, col 1)
print(arr[-1, -1])  # 9 (last element)

# Slicing
print(arr[0:2, 1:3])  # Rows 0-1, cols 1-2: [[2,3],[5,6]]
print(arr[:, 0])      # Toàn bộ column 0: [1, 4, 7]
print(arr[1, :])      # Toàn bộ row 1: [4, 5, 6]
print(arr[::2, ::2])  # Mỗi 2 rows và cols: [[1,3],[7,9]]

# Boolean indexing (rất hữu ích trong ML!)
mask = arr > 5
print(mask)           # [[F,F,F],[F,F,T],[T,T,T]]
print(arr[mask])      # [6, 7, 8, 9]

# Fancy indexing
rows = [0, 2]
cols = [1, 2]
print(arr[rows, cols])  # [arr[0,1], arr[2,2]] = [2, 9]
```

## Vectorized Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Arithmetic operations (element-wise)
print(a + b)    # [6, 8, 10, 12]
print(a * b)    # [5, 12, 21, 32]
print(a ** 2)   # [1, 4, 9, 16]
print(np.sqrt(a))  # [1, 1.414, 1.732, 2]

# Broadcasting - rất quan trọng!
A = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
v = np.array([10, 20, 30])             # shape (3,)
print(A + v)   # [[11,22,33],[14,25,36]] - v được broadcast theo rows

# Thực tế: normalize features
X = np.random.randn(100, 5)  # 100 samples, 5 features
mean = X.mean(axis=0)        # Mean theo columns, shape (5,)
std = X.std(axis=0)          # Std theo columns, shape (5,)
X_normalized = (X - mean) / std  # Broadcasting!

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A @ B)          # Matrix multiplication
print(np.dot(A, B))   # Same as A @ B
print(A.T)            # Transpose
print(np.linalg.inv(A))  # Inverse
print(np.linalg.det(A))  # Determinant
eigenvalues, eigenvectors = np.linalg.eig(A)
```

## Universal Functions (ufuncs)

```python
# Mathematical functions
arr = np.array([-2, -1, 0, 1, 2])
print(np.abs(arr))    # [2, 1, 0, 1, 2]
print(np.exp(arr))    # e^x
print(np.log(np.abs(arr) + 1))  # log(|x| + 1)
print(np.sin(arr))
print(np.maximum(arr, 0))  # ReLU! max(x, 0)

# Aggregation
data = np.random.randn(100, 5)
print(data.sum())           # Tổng tất cả
print(data.sum(axis=0))     # Sum theo columns (shape: 5)
print(data.sum(axis=1))     # Sum theo rows (shape: 100)
print(data.mean(axis=0))    # Mean mỗi column
print(data.std(axis=0))     # Std mỗi column
print(data.min(axis=1))     # Min mỗi row
print(data.argmax(axis=1))  # Index của max mỗi row

# Sorting
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(np.sort(arr))          # [1, 1, 2, 3, 4, 5, 6, 9]
print(np.argsort(arr))       # Indices để sort
print(arr[np.argsort(arr)])  # Same as np.sort
print(np.argpartition(arr, 3)[:3])  # Top 3 smallest (unsorted)
```

## Pandas - Data Analysis Library

```python
import pandas as pd

# Tạo DataFrame
df = pd.DataFrame({
    'name': ['An', 'Bình', 'Châu', 'Dũng', 'Em'],
    'age': [25, 30, 22, 35, 28],
    'city': ['HN', 'HCM', 'HN', 'HCM', 'DN'],
    'salary': [15000000, 25000000, 12000000, None, 18000000],
    'score': [85, 92, 78, 88, 95],
})

print(df)
print(df.shape)    # (5, 5)
print(df.dtypes)   # kiểu dữ liệu mỗi column
print(df.info())   # tổng quan
print(df.describe()) # thống kê cơ bản cho numerical columns
```

## Data Selection và Filtering

```python
# Column selection
print(df['name'])                   # Series
print(df[['name', 'age']])          # DataFrame
print(df.loc[:, 'name':'age'])      # Tương tự

# Row selection
print(df.loc[0])          # Row theo label (index)
print(df.iloc[0])         # Row theo vị trí số
print(df.loc[1:3])        # Rows 1 đến 3 (inclusive trong loc)
print(df.iloc[1:3])       # Rows 1 đến 2 (exclusive trong iloc)

# Filtering
hanoi_mask = df['city'] == 'HN'
print(df[hanoi_mask])     # Chỉ người ở Hà Nội

# Complex filtering
high_salary_hcm = df[(df['city'] == 'HCM') & (df['salary'] > 20_000_000)]
print(high_salary_hcm)

# query() method - dễ đọc hơn
result = df.query("city == 'HN' and age < 30")
print(result)

# isin()
major_cities = df[df['city'].isin(['HN', 'HCM'])]
```

## Data Cleaning

```python
# Tạo DataFrame với missing values
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': ['x', 'y', np.nan, 'y', 'x'],
    'D': [1.0, 2.0, 3.0, 4.0, 5.0],
})

# Kiểm tra missing values
print(data.isnull().sum())     # Số NaN mỗi column
print(data.isnull().any())     # Column nào có NaN?
print(data.notnull().all())    # Column nào không có NaN?

# Xử lý missing values
data_dropped = data.dropna()                     # Xóa rows có NaN
data_filled = data.fillna(0)                     # Điền 0
data_ffill = data.fillna(method='ffill')         # Forward fill
data_bfill = data.fillna(method='bfill')         # Backward fill
data['A'] = data['A'].fillna(data['A'].mean())   # Điền với mean

# Duplicate handling
print(data.duplicated().sum())  # Số rows duplicate
data_clean = data.drop_duplicates()
data_clean = data.drop_duplicates(subset=['C'])  # Theo specific column

# Data type conversion
df['age'] = df['age'].astype(float)
df['score'] = pd.to_numeric(df['score'], errors='coerce')  # Lỗi -> NaN
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
```

## GroupBy và Aggregation

```python
# GroupBy
city_stats = df.groupby('city').agg({
    'age': ['mean', 'min', 'max'],
    'salary': ['mean', 'sum', 'count'],
    'score': 'mean',
})
print(city_stats)

# Sau groupby, có thể reset index
city_avg = df.groupby('city')['salary'].mean().reset_index()
city_avg.columns = ['city', 'avg_salary']

# Nhiều operations
summary = df.groupby('city').agg(
    avg_age=('age', 'mean'),
    total_salary=('salary', 'sum'),
    employee_count=('name', 'count'),
).round(2)

# Apply custom function
def salary_level(salary):
    if pd.isna(salary):
        return 'Unknown'
    elif salary < 15_000_000:
        return 'Low'
    elif salary < 25_000_000:
        return 'Medium'
    else:
        return 'High'

df['salary_level'] = df['salary'].apply(salary_level)
```

## Merge và Join

```python
departments = pd.DataFrame({
    'city': ['HN', 'HCM', 'DN'],
    'department': ['Engineering', 'Sales', 'Marketing'],
    'budget': [5_000_000_000, 8_000_000_000, 3_000_000_000],
})

# Inner join (chỉ lấy city xuất hiện ở cả hai)
merged_inner = df.merge(departments, on='city', how='inner')

# Left join (giữ tất cả từ df bên trái)
merged_left = df.merge(departments, on='city', how='left')

# Outer join (giữ tất cả)
merged_outer = df.merge(departments, on='city', how='outer')

# Merge với điều kiện phức tạp
merged_custom = df.merge(
    departments,
    left_on='city',
    right_on='city',
    suffixes=('_employee', '_dept')
)
```

## Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

def quick_eda(df: pd.DataFrame, target_col: str = None):
    """Thực hiện EDA cơ bản cho DataFrame."""

    print("=" * 60)
    print("BASIC INFO")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    print("\n" + "=" * 60)
    print("NUMERICAL SUMMARY")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include=np.number).columns
    print(df[numeric_cols].describe().round(2))

    print("\n" + "=" * 60)
    print("CATEGORICAL SUMMARY")
    print("=" * 60)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))

    # Visualizations
    fig, axes = plt.subplots(
        len(numeric_cols), 2,
        figsize=(12, 4 * len(numeric_cols))
    )

    for i, col in enumerate(numeric_cols):
        # Histogram
        axes[i, 0].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[i, 0].set_title(f'Distribution: {col}')
        axes[i, 0].set_xlabel(col)
        axes[i, 0].set_ylabel('Count')

        # Box plot
        axes[i, 1].boxplot(df[col].dropna())
        axes[i, 1].set_title(f'Box Plot: {col}')
        axes[i, 1].set_ylabel(col)

    plt.tight_layout()
    plt.show()

    # Correlation matrix
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

# Sử dụng với Iris dataset
from sklearn.datasets import load_iris
iris_data = load_iris()
iris_df = pd.DataFrame(
    iris_data.data,
    columns=iris_data.feature_names
)
iris_df['species'] = [iris_data.target_names[i] for i in iris_data.target]
quick_eda(iris_df, target_col='species')
```

---PRACTICE---

## Bài tập thực hành

### Bài tập 1: NumPy Linear Algebra

Sử dụng NumPy để giải bài toán:
- Tạo ma trận đặc trưng `X` (100 samples, 3 features) ngẫu nhiên
- Tạo vector nhãn `y = 2*x1 - 3*x2 + 1.5*x3 + noise`
- Giải linear regression bằng **Normal Equation**: `w = (X'X)^(-1) X'y`
- So sánh với `np.linalg.lstsq`
- Tính và in ra: MSE, MAE, R² score
- Vẽ actual vs predicted plot

### Bài tập 2: Pandas EDA thực tế

Download dataset Titanic từ URL sau: `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`

Thực hiện EDA:
- Kiểm tra và xử lý missing values
- Phân tích tỷ lệ sống sót theo: Pclass, Sex, Age group, Embarked
- Tạo feature mới: `FamilySize = SibSp + Parch + 1`, `Title` từ Name
- Vẽ visualizations: survival rate by các factors
- Chuẩn bị dataset sạch cho ML (encode categoricals, fill missing)

### Bài tập 3: Sliding Window Feature Engineering

Cho time series data (chuỗi giá cổ phiếu), tạo features:
- Moving averages: MA-5, MA-10, MA-20
- Returns: daily return, 5-day return
- Volatility: rolling standard deviation
- RSI indicator (Relative Strength Index)
- Bollinger Bands
Sử dụng chỉ NumPy và Pandas, không dùng thư viện tài chính.

---SOLUTION---

## Lời giải

### Giải bài tập 1: Linear Algebra với NumPy

```python
# numpy_linear_algebra.py
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Tạo data
n_samples, n_features = 100, 3
X_raw = np.random.randn(n_samples, n_features)

# True coefficients: [2, -3, 1.5]
true_weights = np.array([2.0, -3.0, 1.5])
noise = np.random.randn(n_samples) * 0.5

y = X_raw @ true_weights + noise

# Thêm bias term (column của 1s)
X = np.column_stack([np.ones(n_samples), X_raw])  # (100, 4)

# Normal Equation: w = (X'X)^-1 X'y
# Giải phương trình bình phương tối thiểu
XTX = X.T @ X         # (4, 4)
XTy = X.T @ y         # (4,)
weights_normal = np.linalg.inv(XTX) @ XTy
print(f"Normal Equation weights: {weights_normal}")

# lstsq (ổn định hơn, không cần inverse)
weights_lstsq, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
print(f"lstsq weights: {weights_lstsq}")

# Predictions
y_pred = X @ weights_normal

# Metrics
mse = np.mean((y - y_pred) ** 2)
mae = np.mean(np.abs(y - y_pred))
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\nMSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")

# True weights (bias=0): [2, -3, 1.5]
print(f"\nTrue weights: [0, 2, -3, 1.5]")
print(f"Estimated: {weights_normal.round(4)}")

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.6, edgecolors='none')
min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title(f'Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Giải bài tập 2: Titanic EDA

```python
# titanic_eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# 1. Missing values
# Age: điền với median theo Pclass và Sex
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)

# Embarked: điền với mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Cabin: nhiều quá, tạo feature có/không cabin
df['HasCabin'] = df['Cabin'].notna().astype(int)
df.drop('Cabin', axis=1, inplace=True)

# 2. Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Extract title từ Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_map = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Mme': 'Mrs', 'Countess': 'Rare', 'Lady': 'Rare',
}
df['Title'] = df['Title'].map(title_map).fillna('Rare')

# Age groups
df['AgeGroup'] = pd.cut(df['Age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
)

# 3. Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Survival by Pclass
survival_pclass = df.groupby('Pclass')['Survived'].mean()
axes[0,0].bar(survival_pclass.index, survival_pclass.values, color=['#e74c3c', '#f39c12', '#2ecc71'])
axes[0,0].set_title('Survival Rate by Pclass')
axes[0,0].set_ylabel('Survival Rate')
axes[0,0].set_ylim(0, 1)

# Survival by Sex
survival_sex = df.groupby('Sex')['Survived'].mean()
axes[0,1].bar(survival_sex.index, survival_sex.values, color=['#3498db', '#e91e8c'])
axes[0,1].set_title('Survival Rate by Sex')
axes[0,1].set_ylim(0, 1)

# Age distribution by survival
axes[0,2].hist(df[df['Survived']==1]['Age'].dropna(), bins=20, alpha=0.5, label='Survived', color='green')
axes[0,2].hist(df[df['Survived']==0]['Age'].dropna(), bins=20, alpha=0.5, label='Not survived', color='red')
axes[0,2].set_title('Age Distribution by Survival')
axes[0,2].legend()

# Survival by Title
survival_title = df.groupby('Title')['Survived'].mean().sort_values(ascending=False)
axes[1,0].bar(range(len(survival_title)), survival_title.values)
axes[1,0].set_xticks(range(len(survival_title)))
axes[1,0].set_xticklabels(survival_title.index, rotation=45)
axes[1,0].set_title('Survival Rate by Title')
axes[1,0].set_ylim(0, 1)

# Family size vs survival
survival_family = df.groupby('FamilySize')['Survived'].mean()
axes[1,1].plot(survival_family.index, survival_family.values, 'o-', color='purple')
axes[1,1].set_title('Survival Rate by Family Size')
axes[1,1].grid(True, alpha=0.3)

# Correlation heatmap
numeric_df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'HasCabin']].dropna()
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,2], center=0)
axes[1,2].set_title('Correlation Heatmap')

plt.suptitle('Titanic EDA', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 4. Chuẩn bị dataset cho ML
df_ml = df.copy()

# Encode categoricals
df_ml['Sex'] = (df_ml['Sex'] == 'female').astype(int)
df_ml = pd.get_dummies(df_ml, columns=['Embarked', 'Title', 'AgeGroup'], drop_first=True)
df_ml['LogFare'] = np.log1p(df_ml['Fare'])

# Drop columns không cần thiết
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Fare']
df_ml = df_ml.drop(columns=cols_to_drop)

print(f"\nDataset sau khi chuẩn bị: {df_ml.shape}")
print(f"Missing values còn lại: {df_ml.isnull().sum().sum()}")
print(df_ml.head())
```

### Giải bài tập 3: Time Series Feature Engineering

```python
# feature_engineering.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_stock_data(n_days: int = 252) -> pd.DataFrame:
    """Tạo dữ liệu giá cổ phiếu giả."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    dates = pd.date_range(start='2023-01-01', periods=n_days + 1, freq='B')
    volume = np.random.randint(1_000_000, 10_000_000, n_days + 1)

    return pd.DataFrame({'close': prices, 'volume': volume}, index=dates)

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm tất cả technical features."""
    df = df.copy()

    # Moving averages
    for window in [5, 10, 20]:
        df[f'MA_{window}'] = df['close'].rolling(window).mean()

    # Returns
    df['daily_return'] = df['close'].pct_change()
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)

    # Volatility
    df['volatility_10d'] = df['daily_return'].rolling(10).std()
    df['volatility_20d'] = df['daily_return'].rolling(20).std()

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_window = 20
    ma = df['close'].rolling(bb_window).mean()
    std = df['close'].rolling(bb_window).std()
    df['BB_upper'] = ma + 2 * std
    df['BB_lower'] = ma - 2 * std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / ma
    df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Volume features
    df['volume_MA5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_MA5']

    return df

# Sử dụng
df = generate_stock_data()
df_features = add_technical_features(df)
df_features.dropna(inplace=True)

print(f"Features tạo được: {df_features.shape[1]} columns")
print(df_features.tail())

# Visualize
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Price + MAs + BB
axes[0].plot(df_features['close'], label='Close', linewidth=1.5)
for ma in ['MA_5', 'MA_10', 'MA_20']:
    axes[0].plot(df_features[ma], label=ma, linewidth=1, alpha=0.7)
axes[0].fill_between(df_features.index, df_features['BB_upper'], df_features['BB_lower'], alpha=0.1, color='gray')
axes[0].set_title('Price, Moving Averages & Bollinger Bands')
axes[0].legend(fontsize=8)

# Returns
axes[1].bar(df_features.index, df_features['daily_return'], alpha=0.7, color='steelblue', width=1)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_title('Daily Returns')

# RSI
axes[2].plot(df_features['RSI_14'], color='purple')
axes[2].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
axes[2].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
axes[2].set_ylim(0, 100)
axes[2].set_title('RSI (14)')
axes[2].legend()

# MACD
axes[3].plot(df_features['MACD'], label='MACD', color='blue')
axes[3].plot(df_features['MACD_signal'], label='Signal', color='red')
axes[3].bar(df_features.index, df_features['MACD_hist'], alpha=0.5, color='gray', width=1)
axes[3].axhline(y=0, color='black', linewidth=0.5)
axes[3].set_title('MACD')
axes[3].legend()

plt.tight_layout()
plt.show()
```
