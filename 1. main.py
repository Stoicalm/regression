import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# 1. 读取数据
file_path = "回归预测.xlsx"
train_df = pd.read_excel(file_path, sheet_name="训练集", header=None)
test_df = pd.read_excel(file_path, sheet_name="测试集", header=None)

# 2. 分离特征和目标
# 前30列为数值特征（索引0-29），第31列为药物类型（索引30），第32列为目标（索引31）
X_train_raw = train_df.iloc[:, :30]
cat_train = train_df.iloc[:, 30].astype(str)  # 转换为字符串
y_train = train_df.iloc[:, 31]

X_test_raw = test_df.iloc[:, :30]
cat_test = test_df.iloc[:, 30].astype(str)
y_test = test_df.iloc[:, 31]

# 3. 构建预处理管道：数值特征保留，分类特征独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', list(range(30))),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [30])
    ])

# 将分类特征作为额外列加入
X_train = np.hstack([X_train_raw.values, cat_train.values.reshape(-1, 1)])
X_test = np.hstack([X_test_raw.values, cat_test.values.reshape(-1, 1)])

# 4. 定义模型和超参数网格
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 5. 网格搜索交叉验证
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(preprocessor.fit_transform(X_train), y_train)
best_model = grid_search.best_estimator_

# 6. 在测试集上预测
X_test_processed = preprocessor.transform(X_test)
y_pred = best_model.predict(X_test_processed)

# 7. 计算误差
abs_errors = np.abs(y_pred - y_test)
rel_errors = abs_errors / (y_test)
s_errors = (np.abs(y_pred - y_test))**2

mean_abs = np.mean(abs_errors)
var_abs = np.var(abs_errors)
mean_rel = np.mean(rel_errors)
var_rel = np.var(rel_errors)
mean_s = np.mean(s_errors)
var_s = np.var(s_errors)

# 8. 输出结果
print("最佳超参数:", grid_search.best_params_)
print("\n测试集误差统计:")
print(f"绝对误差均值: {mean_abs:.4f}")
print(f"绝对误差方差: {var_abs:.4f}")
print(f"相对误差均值: {mean_rel:.4f}")
print(f"相对误差方差: {var_rel:.4f}")
print(f"平方误差均值: {mean_s:.4f}")
print(f"平方误差方差: {var_s:.4f}")

results = pd.DataFrame({
    '真实值': y_test.values,
    '预测值': y_pred,
    '绝对误差': abs_errors,
    '相对误差': rel_errors,
    '平方误差': s_errors  
})
print("\n前10个测试样本的预测详情:")
print(tabulate(results.head(10), headers='keys', tablefmt='grid', showindex=False))

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True value')
plt.ylabel('Predicted value')
plt.title('True value vs Predicted value')
plt.grid(True)
plt.show()

# ===================== 关键修改：残差分布图的红线改为残差均值 =====================
residuals = y_test - y_pred
mean_residuals = np.mean(residuals)  # 新增：计算残差的均值
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color='blue', alpha=0.7)
# 修改：红线位置从0改为残差均值，并添加均值标注的图例
plt.axvline(mean_residuals, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_residuals:.4f}')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.legend()  # 新增：显示图例（和平方误差图保持一致）
plt.grid(True)
plt.show()
# ================================================================================

plt.figure(figsize=(10, 6))
sns.histplot(s_errors, kde=True, bins=30, color='orange', alpha=0.7)
plt.axvline(mean_s, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_s:.4f}')
plt.xlabel('Squared Error')
plt.ylabel('Frequency')
plt.title('Squared Error Distribution')
plt.legend()
plt.grid(True)
plt.show()

print("最佳超参数:", grid_search.best_params_)