# 回归（Regressão）复习笔记 — G22

## 问题定义
- **目标**：根据学生行为数据预测 `exam_score`（连续值，0~100）
- **输入特征**：年龄、性别、学习时间、睡眠时间、游戏时间、咖啡因摄入等行为变量
- **排除**：`student_id`、`productivity_score`、`burnout_level`、`focus_index`（直接与目标相关，会造成数据泄露）

---

## 1. 数据预处理

### 缺失值处理
- **数值列** → 用 **中位数（median）** 填充（不用均值，因为均值对异常值敏感）
- **类别列** → 用 **众数（mode）** 填充
- 关键：先 fit X_learning，再用同样的值 transform X_test（防止数据泄露）

```python
medians = X_learning[num_cols].median()
modes   = X_learning[cat_cols].mode().iloc[0]
X_learning[num_cols] = X_learning[num_cols].fillna(medians)
X_test[num_cols]     = X_test[num_cols].fillna(medians)  # 用训练集的值！
```

### 编码（Encoding）
- 类别特征（gender、academic_level、internet_quality）→ **One-Hot Encoding**
- 为什么不用 Ordinal？这三个特征没有内在顺序关系，Ordinal 会引入虚假的大小关系

### Train/Test Split
- `test_size=0.2`，`random_state=7`（保证可复现）

---

## 2. Feature Selection（特征选择）— 5种方法

所有方法都基于 X_learning，绝不能碰 X_test（防止泄露）。

### 方法A：Variance Threshold（方差阈值）
- **原理**：方差极小的特征几乎不变化 → 信息量低 → 丢掉
- **代码**：`VarianceThreshold(threshold=0.01)`
- **局限**：方差低不代表一定没用，只是最简单的过滤

### 方法B：Correlation Filter（相关性过滤）
- **原理**：两个特征高度相关（>0.9）→ 信息重复 → 保留一个
- 只看特征之间的相关性，不考虑与 y 的关系
- **局限**：只检测线性关系，无法发现非线性冗余

### 方法C：Mutual Information（互信息）
- **原理**：衡量特征与 y 之间的统计依赖关系，非线性也能捕捉
- 回归用 `mutual_info_regression`（分类用 `mutual_info_classif`）
- 取 top-10 得分最高的特征

### 方法D：Sequential Feature Selection（顺序特征选择）
- **原理**：贪心算法，每次加入让模型 RMSE 最小的特征，直到选够 8 个
- 属于 **Wrapper** 方法：把模型包在里面评估特征
- 用 `LinearRegression` 作为评估器（中性、快速）
- **局限**：贪心，不保证全局最优

### 方法E：PCA（主成分分析）
- **原理**：不选原有特征，而是创造新的主成分（原特征的线性组合）
- 选取能解释 95% 方差的最少主成分数量
- **注意**：PCA 之前必须 StandardScaler！
- **优点**：降维效果好；**缺点**：主成分无法解释（失去原始特征含义）

### 评估函数 `comparar_feature_sets`
每个模型都用这个函数对 6 个特征集（Baseline + 5种方法）打分：
- 统一用 StandardScaler 缩放
- 用 clone(model) 保证每次用全新模型
- 返回按 RMSE 排序的表格

---

## 3. 模型详解

### 3.1 Decision Tree Regressor（决策树回归）

**工作原理**：
- 每个节点选一个特征和一个分割点，使两个子集的 `squared_error` 之和最小
- 叶节点的预测值 = 该叶节点训练样本的 y 均值
- 树越深 → 对训练数据拟合越好 → 越容易过拟合

**GridSearchCV 参数**：
```python
param_grid = {
    "criterion": ["squared_error", "absolute_error"],  # 分裂标准
    "max_depth": [1, 2, 3, 4, 5, 6, None],             # 最大深度
    "min_samples_split": [2, 5, 10, 20],               # 分裂最少样本数
    "min_samples_leaf": [1, 2, 5, 10],                 # 叶节点最少样本数
}
```

**为什么不太需要 feature selection**：
DT 在每个节点自动选最有用的特征，无用特征不会被选中分裂，天生有筛选机制。

**评估指标**：RMSE、MAE、R²

---

### 3.2 Linear Regression（线性回归）

**工作原理**：
- `y = α + β₁x₁ + β₂x₂ + ... + βₙxₙ`
- 通过最小二乘法找到使残差平方和最小的系数
- 每个 β 表示该特征对 y 的影响大小

**必须 StandardScaler 的原因**：
- 不同特征尺度差异大（study_hours 0~24，caffeine_intake 0~600）
- 不缩放的话，大尺度特征会主导系数，小尺度特征被忽视

**为什么非常需要 feature selection**：
- 高度相关的特征（multicollinearity）会让系数不稳定
- 噪声特征直接增加 RMSE

---

### 3.3 KNN Regressor（K近邻回归）

**工作原理**：
- 找测试样本在训练集中最近的 K 个邻居
- 预测值 = K 个邻居的 y 值的平均（或加权平均）
- 没有显式训练，所有计算在预测时完成

**GridSearchCV 参数**：
```python
param_grid_knn = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],           # K值
    "weights": ["uniform", "distance"],              # uniform=等权；distance=距离越近权重越大
    "metric": ["euclidean", "manhattan"],            # 距离计算方式
}
```

**K 的影响**：
- K 小 → 模型复杂，对噪声敏感，过拟合
- K 大 → 模型简单，边界平滑，欠拟合

**为什么最需要 feature selection（维度诅咒）**：
- 特征越多，所有点之间的距离趋向相等
- 距离失去意义，KNN 完全失效
- PCA 对 KNN 特别有效

---

### 3.4 Random Forest Regressor（随机森林回归）

**工作原理**：
- 训练 N 棵决策树（`n_estimators`），每棵树：
  - 用 **Bootstrap 采样**（有放回抽样）的训练子集
  - 每次分裂只考虑随机选出的特征子集（`max_features`）
- 预测 = N 棵树的平均

**GridSearchCV 参数**：
```python
param_grid_rf = {
    "n_estimators": [100, 200],          # 树的数量
    "max_depth": [5, 10, None],          # 每棵树的最大深度
    "min_samples_split": [2, 5, 10],     # 分裂最少样本数
}
```

**为什么比 DT 强**：
- 多棵树的平均降低了方差（偏差-方差权衡）
- 随机特征选择让噪声特征被稀释

**为什么不太需要 feature selection**：
随机特征采样本身就是一种内置的特征筛选机制。

---

### 3.5 MLP Regressor（多层感知机回归）

**工作原理**：
- 输入层 → 隐藏层（非线性激活函数）→ 输出层（1个神经元，直接输出数值）
- 使用反向传播（backpropagation）和梯度下降更新权重
- `hidden_layer_sizes=(64, 32)` = 两个隐藏层，分别有 64 和 32 个神经元

**GridSearchCV 参数**：
```python
param_grid_mlp = {
    "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],  # 网络结构
    "activation": ["relu", "tanh"],                        # 激活函数
    "alpha": [0.0001, 0.001],                              # L2正则化系数
}
```

**ReLU vs tanh**：
- ReLU：`max(0, x)`，计算快，不会梯度消失
- tanh：`(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)`，输出在 [-1,1]，对称

---

## 4. 评估指标

| 指标 | 公式 | 方向 | 含义 |
|---|---|---|---|
| RMSE | √(Σ(ŷ-y)²/n) | 越小越好 | 对大误差更敏感 |
| MAE | Σ|ŷ-y|/n | 越小越好 | 所有误差等权 |
| R² | 1 - SS_res/SS_tot | 越接近1越好 | 模型解释了多少方差 |

**我们的主要指标**：RMSE（因为大误差对学生预测来说影响更严重）

---

## 5. 错误分析（Análise de Erro）

- 计算残差：`residuo = predito - real`
- 找 top-10 最差预测（|残差| 最大）
- 比较 outlier 样本 vs 普通样本的特征均值
- **结论**：
  - 过高预测（sobreavaliação）：`study_hours` 高但 `mental_health_score` 低
  - 过低预测（subestimação）：`gaming_hours` 高但实际成绩好

---

## 6. SHAP（可解释性分析）

**什么是 SHAP**：
- 基于博弈论（Shapley values）
- 衡量每个特征对某次预测的贡献（+/-）
- **TreeExplainer**：专门针对树模型（RF、DT），速度快

**4个例子的选择逻辑**（回归版 TP/TN/FP/FN）：
| 类型 | 选法 |
|---|---|
| Acerto alto（TP equiv）| 残差小 + 实际分数高 |
| Acerto baixo（TN equiv）| 残差小 + 实际分数低 |
| Sobreavaliação（FP equiv）| 预测 >> 实际（残差大正值）|
| Subestimação（FN equiv）| 预测 << 实际（残差大负值）|

**RF vs DT 的差异**：
- RF：把重要性分散给更多特征（ensemble 效果）
- DT：把重要性集中在少数主要特征（路径固定）

---

## 7. 可能被问到的问题 & 答案

**Q: 为什么用 RMSE 不用 MAE？**
A: RMSE 对大误差惩罚更重（平方），在教育预测中，严重偏差比平均偏差更重要。

**Q: 为什么 KNN 需要 StandardScaler？**
A: KNN 计算欧氏距离，特征尺度不同会让大尺度特征主导距离计算，小尺度特征被忽略。

**Q: PCA 和 feature selection 有什么区别？**
A: Feature selection 保留原始特征的子集；PCA 创造全新的主成分（线性组合），失去原始特征的可解释性。

**Q: GridSearchCV 的 CV 是什么？**
A: K-Fold Cross-Validation。把 X_learning 分成 K 份，轮流用 K-1 份训练、1 份验证，取平均分数，防止过拟合到特定划分。

**Q: Random Forest 为什么比单棵 DT 好？**
A: Bagging（Bootstrap aggregating）：多棵树的平均降低了方差。每棵树用不同的训练子集，最终预测更稳健。

**Q: 为什么 Linear Regression 的系数需要 StandardScaler 才有可比性？**
A: 缩放后，所有特征系数都在相同尺度上，大系数 = 真正重要，不是因为特征本身尺度大。

**Q: SFS 为什么用 LinearRegression 作为 base estimator？**
A: 用简单中性的模型来评估特征，避免复杂模型（如RF）掩盖特征质量差异。目的是找"普遍有用的特征"，不是"RF专用特征"。
