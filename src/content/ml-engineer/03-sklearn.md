# Scikit-learn

## Scikit-learn là gì?

**Scikit-learn** (sklearn) là thư viện Machine Learning mã nguồn mở phổ biến nhất cho Python. Nó cung cấp các implementation đơn giản, hiệu quả của hàng trăm thuật toán ML với API nhất quán.

### Triết lý thiết kế

Mọi model trong sklearn đều tuân theo cùng một interface:
```python
model = SomeAlgorithm(hyperparameters)
model.fit(X_train, y_train)     # Train
predictions = model.predict(X_test)  # Predict
score = model.score(X_test, y_test)  # Evaluate
```

## Linear Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Tạo data
np.random.seed(42)
X = np.random.randn(200, 3)
y = 2 * X[:, 0] - 3 * X[:, 1] + 1.5 * X[:, 2] + np.random.randn(200) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(f"Coefficients: {lr.coef_}")    # [2, -3, 1.5] - khớp với true values
print(f"Intercept: {lr.intercept_:.4f}")
print(f"R² test: {r2_score(y_test, y_pred):.4f}")
print(f"MSE test: {mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE test: {mean_absolute_error(y_test, y_pred):.4f}")

# Ridge Regression (L2 regularization) - chống overfitting
ridge = Ridge(alpha=1.0)  # alpha là strength của regularization
ridge.fit(X_train, y_train)
print(f"\nRidge R²: {ridge.score(X_test, y_test):.4f}")

# Lasso Regression (L1 regularization) - feature selection
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"Lasso R²: {lasso.score(X_test, y_test):.4f}")
print(f"Lasso coefficients (sparse): {lasso.coef_}")  # Một số = 0 (feature selection)
```

## Logistic Regression (Classification)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score, roc_curve)

# Binary classification: Cancer detection
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling quan trọng cho logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # KHÔNG fit lại trên test set!

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['Malignant', 'Benign']))

print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")
# [[TN, FP],
#  [FN, TP]]

# Multi-class classification
X_iris, y_iris = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

log_reg_multi = LogisticRegression(
    multi_class='multinomial',  # Softmax cho multi-class
    solver='lbfgs',
    max_iter=1000
)
log_reg_multi.fit(X_tr, y_tr)
print(f"\nMulti-class accuracy: {log_reg_multi.score(X_te, y_te):.4f}")
```

## Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
dt = DecisionTreeClassifier(
    max_depth=3,           # Giới hạn độ sâu để tránh overfitting
    min_samples_split=10,  # Số samples tối thiểu để split
    min_samples_leaf=5,    # Số samples tối thiểu ở leaf
    criterion='gini',      # 'gini' hoặc 'entropy'
    random_state=42
)
dt.fit(X_train, y_train)
print(f"Decision Tree accuracy: {dt.score(X_test, y_test):.4f}")

# Feature importance
feature_names = load_iris().feature_names
for name, importance in zip(feature_names, dt.feature_importances_):
    print(f"  {name}: {importance:.4f}")

# Visualize tree
plt.figure(figsize=(15, 8))
plot_tree(dt, feature_names=feature_names,
          class_names=load_iris().target_names,
          filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.show()

# Overfitting demo
depths = range(1, 15)
train_scores, test_scores = [], []

for depth in depths:
    dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_temp.fit(X_train, y_train)
    train_scores.append(dt_temp.score(X_train, y_train))
    test_scores.append(dt_temp.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'b-o', label='Train score')
plt.plot(depths, test_scores, 'r-o', label='Test score')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Overfitting Demo: Decision Tree Depth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest: ensemble của nhiều decision trees
rf = RandomForestClassifier(
    n_estimators=100,      # Số trees
    max_depth=5,           # Độ sâu tối đa mỗi tree
    max_features='sqrt',   # Số features mỗi split (default: sqrt(n_features))
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,             # Dùng tất cả CPU cores
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"Random Forest accuracy: {rf.score(X_test, y_test):.4f}")
print(classification_report(y_test, y_pred))

# Feature importance (built-in)
feature_importances = pd.DataFrame({
    'feature': load_breast_cancer().feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(
    feature_importances['feature'][:15],
    feature_importances['importance'][:15]
)
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.show()
```

## Cross-Validation

```python
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                       GridSearchCV, RandomizedSearchCV,
                                       learning_curve)

X, y = load_iris(return_X_y=True)

# K-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"CV Scores: {scores}")
print(f"Mean ± Std: {scores.mean():.4f} ± {scores.std():.4f}")

# GridSearchCV - Tìm hyperparameters tốt nhất
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_leaf': [1, 2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X, y)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
best_model = grid_search.best_estimator_

# RandomizedSearchCV - Nhanh hơn GridSearch khi space lớn
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.3, 0.7),
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,   # Số combinations cần thử
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
)
random_search.fit(X, y)
print(f"\nRandomized search best: {random_search.best_params_}")

# Learning Curve - Kiểm tra overfitting/underfitting
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'b-o', label='Train score')
plt.fill_between(train_sizes,
                  train_scores.mean(axis=1) - train_scores.std(axis=1),
                  train_scores.mean(axis=1) + train_scores.std(axis=1),
                  alpha=0.1, color='blue')
plt.plot(train_sizes, val_scores.mean(axis=1), 'r-o', label='Validation score')
plt.fill_between(train_sizes,
                  val_scores.mean(axis=1) - val_scores.std(axis=1),
                  val_scores.mean(axis=1) + val_scores.std(axis=1),
                  alpha=0.1, color='red')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Clustering

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

# K-Means
X_cluster = np.random.randn(300, 2)
X_cluster = np.vstack([
    X_cluster[:100] + [0, 0],
    X_cluster[100:200] + [5, 5],
    X_cluster[200:] + [10, 0],
])

# Elbow method để chọn k
inertias = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method')
plt.grid(True, alpha=0.3)
plt.show()

# Với k=3
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X_cluster)
print(f"Silhouette Score: {silhouette_score(X_cluster, labels):.4f}")
# Silhouette: -1 (xấu) đến 1 (tốt), > 0.5 là tốt

# DBSCAN - Không cần biết trước số clusters
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X_cluster)
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = np.sum(labels_db == -1)
print(f"DBSCAN clusters: {n_clusters}, noise points: {n_noise}")
```

## Pipeline

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Simple pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
])

pipeline.fit(X_train, y_train)
print(f"Pipeline accuracy: {pipeline.score(X_test, y_test):.4f}")

# Complex pipeline với ColumnTransformer
numeric_features = ['age', 'salary']
categorical_features = ['city', 'department']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42)),
])
```

---PRACTICE---

## Bài tập thực hành

### Bài tập 1: Complete ML Project

Sử dụng dataset House Prices (từ sklearn hoặc tạo synthetic):
- EDA cơ bản
- Feature engineering: tạo ít nhất 3 features mới
- Thử nghiệm ít nhất 4 models: Linear Regression, Ridge, Random Forest, Gradient Boosting
- GridSearchCV cho model tốt nhất
- Đánh giá trên test set: RMSE, R², MAE
- Visualize: learning curves, feature importance, actual vs predicted

### Bài tập 2: Customer Churn Prediction

Dataset: Tạo synthetic customer data với features:
- tenure (months), monthly_charges, total_charges, num_products, support_calls, churn (binary)

Yêu cầu:
- Xử lý class imbalance (dùng `class_weight='balanced'` hoặc SMOTE)
- Pipeline đầy đủ với preprocessing
- So sánh ít nhất 3 classifiers
- Focus vào Recall (quan trọng hơn Precision trong churn detection)
- ROC curve cho tất cả models

### Bài tập 3: Clustering cho Customer Segmentation

- Tạo dataset khách hàng: recency (ngày mua gần nhất), frequency (số lần mua), monetary (tổng chi tiêu) - RFM model
- K-Means clustering với Elbow method và Silhouette analysis
- Visualize segments
- Mô tả từng segment (High Value, At Risk, etc.)

---SOLUTION---

## Lời giải

### Giải bài tập 1: Complete ML Project

```python
# house_price_ml.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseVal')

print(f"Dataset shape: {X.shape}")
print(f"\nBasic stats:\n{X.describe().round(2)}")

# Feature Engineering
X_fe = X.copy()
X_fe['RoomsPerHouse'] = X['AveRooms'] / X['AveBedrms']
X_fe['PopulationPerHouse'] = X['Population'] / X['HouseAge']
X_fe['BedroomsRatio'] = X['AveBedrms'] / X['AveRooms']
X_fe['IncomePerRoom'] = X['MedInc'] / X['AveRooms']

# Log transform skewed features
X_fe['log_MedInc'] = np.log1p(X_fe['MedInc'])
X_fe['log_Population'] = np.log1p(X_fe['Population'])

X_train, X_test, y_train, y_test = train_test_split(
    X_fe, y, test_size=0.2, random_state=42
)

# Define models
models = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae, 'model': model, 'pred': y_pred}
    print(f"{name}: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")

# Best model: GridSearchCV trên Gradient Boosting
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
}
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid, cv=3, scoring='neg_rmse', n_jobs=-1, verbose=0
)
gb_grid.fit(X_train, y_train)
print(f"\nBest GB params: {gb_grid.best_params_}")

best_pred = gb_grid.best_estimator_.predict(X_test)
print(f"Best GB RMSE: {np.sqrt(mean_squared_error(y_test, best_pred)):.4f}")

# Visualize results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, (name, res) in enumerate(results.items()):
    axes[i].scatter(y_test, res['pred'], alpha=0.3, s=5)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[i].set_title(f"{name}\nR²={res['R2']:.3f}")
    axes[i].set_xlabel('Actual')
    axes[i].set_ylabel('Predicted')

plt.tight_layout()
plt.show()
```

### Giải bài tập 2: Customer Churn Prediction

```python
# churn_prediction.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, roc_curve, auc,
                              confusion_matrix, recall_score)
import matplotlib.pyplot as plt

# Generate synthetic churn data
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'tenure': np.random.randint(1, 72, n),
    'monthly_charges': np.random.uniform(20, 100, n),
    'num_products': np.random.randint(1, 6, n),
    'support_calls': np.random.poisson(2, n),
})
data['total_charges'] = data['tenure'] * data['monthly_charges'] * np.random.uniform(0.8, 1.2, n)

# Churn probability based on features (realistic)
churn_prob = (
    0.3 +
    -0.005 * data['tenure'] +
    0.003 * data['monthly_charges'] +
    -0.05 * data['num_products'] +
    0.08 * data['support_calls']
)
churn_prob = churn_prob.clip(0.05, 0.95)
data['churn'] = (np.random.random(n) < churn_prob).astype(int)

print(f"Churn rate: {data['churn'].mean():.2%}")
print(f"Class distribution:\n{data['churn'].value_counts()}")

X = data.drop('churn', axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
}

# ROC curve plot
plt.figure(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    recall = recall_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f}, Recall={recall:.3f})')
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curves - Churn Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Giải bài tập 3: Customer Segmentation

```python
# customer_segmentation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate RFM data
np.random.seed(42)
n_customers = 500

# 4 segments: VIP, Regular, At-Risk, Inactive
segments = np.random.choice(['VIP', 'Regular', 'AtRisk', 'Inactive'],
                             size=n_customers, p=[0.1, 0.4, 0.3, 0.2])

rfm_data = []
for seg in segments:
    if seg == 'VIP':
        r = np.random.randint(1, 30)
        f = np.random.randint(20, 50)
        m = np.random.uniform(5000, 20000)
    elif seg == 'Regular':
        r = np.random.randint(20, 90)
        f = np.random.randint(5, 20)
        m = np.random.uniform(500, 5000)
    elif seg == 'AtRisk':
        r = np.random.randint(90, 180)
        f = np.random.randint(3, 15)
        m = np.random.uniform(200, 2000)
    else:  # Inactive
        r = np.random.randint(180, 365)
        f = np.random.randint(1, 5)
        m = np.random.uniform(50, 500)
    rfm_data.append({'Recency': r, 'Frequency': f, 'Monetary': m, 'TrueSegment': seg})

df_rfm = pd.DataFrame(rfm_data)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_rfm[['Recency', 'Frequency', 'Monetary']])

# Elbow method
inertias = []
silhouettes = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(k_range, inertias, 'bo-')
axes[0].axvline(x=4, color='red', linestyle='--', alpha=0.7)
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouettes, 'ro-')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# K=4 clusters
km = KMeans(n_clusters=4, random_state=42, n_init=10)
df_rfm['Cluster'] = km.fit_predict(X_scaled)

# Analyze clusters
cluster_stats = df_rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
print("Cluster Statistics:")
print(cluster_stats)

# Label clusters
labels = {0: 'High Value', 1: 'At Risk', 2: 'Regular', 3: 'Inactive'}
df_rfm['SegmentName'] = df_rfm['Cluster'].map(labels)
print(f"\nCluster sizes:\n{df_rfm['SegmentName'].value_counts()}")

# 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['gold', 'red', 'blue', 'gray']
for cluster_id in range(4):
    mask = df_rfm['Cluster'] == cluster_id
    ax.scatter(
        df_rfm[mask]['Recency'],
        df_rfm[mask]['Frequency'],
        df_rfm[mask]['Monetary'],
        c=colors[cluster_id],
        label=labels[cluster_id],
        alpha=0.6, s=30
    )
ax.set_xlabel('Recency (days)')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary ($)')
ax.set_title('Customer Segmentation (RFM)')
ax.legend()
plt.show()
```
