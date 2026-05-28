# 分类（Classificação）复习笔记 — G22

## 问题定义
- **目标**：把 `exam_score` 转化成二分类标签，预测学生是否通过
- **分类方案**：
  - 多类：Insuficiente / Suficiente / Bom / Excelente
  - 二值（最终用这个）：0 = Insuficiente（不及格），1 = Suficiente+（及格）
- **输入特征**：与回归相同的行为变量
- **排除**：`student_id`、`productivity_score`、`burnout_level`、`focus_index`

---

## 1. 数据预处理

### exam_score → 二值标签
```python
# Insuficiente = 0，其余 = 1
categories = [["Insuficiente", "Suficiente", "Bom", "Excelente"]]
# 用 OrdinalEncoder 转成数字，再二值化
y = (encoded_score > 0).astype(int)  # 0=不及格，1=及格
```

### 缺失值、编码、Split
- 与回归相同：中位数填数值、众数填类别、One-Hot Encoding
- **Split 用 Stratified**（保证正负样本比例在 train/test 中一致）

### 评估指标
| 指标 | 含义 | 方向 |
|---|---|---|
| Accuracy | 正确预测的比例 | 越大越好 |
| F1-Score | Precision 和 Recall 的调和平均 | 越大越好 |
| ROC-AUC | 分类能力综合指标 | 越大越好 |

**主要指标：F1-Score**（因为正负样本可能不平衡，Accuracy 会误导）

---

## 2. Feature Selection（与回归相同的5种方法）

**区别**：
- Mutual Information 用 `mutual_info_classif`（不是 regression）
- SFS 用 `LogisticRegression` 作为 base estimator（不是 LinearRegression）
- 评估用 Accuracy 或 F1（不是 RMSE）

---

## 3. 模型详解

### 3.1 Decision Tree Classifier（决策树分类）

**工作原理**：
- 每个节点选特征，使子集的 **Gini 不纯度** 或 **信息熵** 最小
- Gini = `1 - Σpᵢ²`（越低越纯）
- 叶节点的预测 = 该叶节点中多数类的标签

**与回归版 DT 的区别**：
- 分类：criterion = `gini` 或 `entropy`
- 回归：criterion = `squared_error` 或 `absolute_error`

**GridSearchCV 参数**（与回归类似）：
```python
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [...],
    "min_samples_split": [...],
    "min_samples_leaf": [...],
}
```

---

### 3.2 KNN Classifier（K近邻分类）

**工作原理**：
- 找最近 K 个邻居
- 预测 = K 个邻居中**多数类**（投票）
- 与回归版区别：回归取均值，分类取多数票

**各种特征集的效果**：
- **PCA 最好**：KNN 受维度诅咒影响最大，PCA 降维后距离计算更有意义
- SFS 次之

---

### 3.3 Gaussian Naive Bayes（高斯朴素贝叶斯）

**工作原理**：
- 基于贝叶斯定理：`P(y|X) ∝ P(X|y) × P(y)`
- **"朴素"假设**：所有特征之间相互独立（条件独立性）
- **"高斯"假设**：每个特征在给定类别下服从正态分布
- 不需要 GridSearchCV（几乎没有超参数）

**公式**：
```
P(x_i | y) = (1/√(2πσ²)) × exp(-(x_i - μ)²/(2σ²))
```
μ 和 σ 分别是类别 y 下特征 x_i 的均值和标准差。

**优点**：速度极快，小数据效果好
**缺点**：独立性假设在现实中几乎不成立，但实践中效果还可以

---

### 3.4 Random Forest Classifier（随机森林分类）

**工作原理**：
- 与回归版相同，但叶节点用多数投票而非均值
- `predict_proba()` 返回属于每个类别的概率

---

### 3.5 MLP Classifier（多层感知机分类）

**工作原理**：
- 与回归版相同，但输出层：
  - 二分类：1个神经元 + Sigmoid 激活 → 输出概率
  - 多分类：N个神经元 + Softmax → 输出每类概率
- 损失函数：cross-entropy（交叉熵）

---

## 4. 模型对比总表

对比了所有模型在不同 feature set 下的 F1-Score、Accuracy、ROC-AUC：

| 模型 | 最佳 Feature Set | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|
| SFS | Sequential | 最高 | ~ | ~ |
| MI | Mutual Info | 次之 | ~ | ~ |
| PCA | PCA(n=9) | ~ | ~ | ~ |
| Variance | Variance | ~ | ~ | ~ |
| Correlation | Correlation | 最低 | ~ | ~ |

---

## 5. Clustering（非监督学习）

**目标**：不用标签，找学生群体的自然分组（Personas）

**K-Means 算法**：
1. 随机初始化 K 个中心点
2. 每个样本分配给最近的中心
3. 重新计算每个簇的均值为新中心
4. 重复直到收敛

**选 K 的方法**：
- **Elbow Method（肘部法）**：画 inertia（簇内距离平方和）vs K，找"弯折点"
- **Silhouette Score（轮廓系数）**：衡量样本与自己簇的相似度 vs 与最近其他簇的相似度，越接近 1 越好

**结果**：K=4 最佳（Silhouette Score 最高）

**4个 Personas**：
- Persona 0：自律专注型（study 时间多，sleep 好，mental health 好）
- Persona 1：社交娱乐型（social media 多，gaming 多，成绩一般）
- Persona 2：疲惫低效型（sleep 少，caffeine 多，burnout 高）
- Persona 3：平衡发展型（各指标均衡）

---

## 6. SHAP（可解释性）

**与回归版的区别**：
- 分类用 `shap_values = explainer.shap_values(X)` 返回一个 list（每个类别一个矩阵）
- 提取类别 1（及格）的 SHAP 矩阵：`shap_values[1]`

**4个例子**（真正的 TP/TN/FP/FN）：
| 类型 | 含义 |
|---|---|
| TP | 真正例：模型预测及格，实际也及格 |
| TN | 真负例：模型预测不及格，实际也不及格 |
| FP | 假正例：模型预测及格，实际不及格（误判） |
| FN | 假负例：模型预测不及格，实际及格（漏判） |

---

## 7. 可能被问到的问题 & 答案

**Q: 为什么用 F1 不用 Accuracy？**
A: 如果正负样本不平衡，Accuracy 会偏向多数类。F1 综合了 Precision（预测及格的有多少真的及格）和 Recall（实际及格的有多少被预测出来），更公平。

**Q: Naive Bayes 的"朴素"是什么意思？**
A: 假设所有特征之间条件独立——给定类别 y，特征 x₁ 和 x₂ 之间没有关联。现实中这几乎不成立，但计算效率高，效果往往还可以。

**Q: ROC-AUC 是什么？**
A: ROC 曲线：以不同阈值下的 TPR（True Positive Rate）vs FPR（False Positive Rate）画出。AUC 是曲线下面积，1.0 = 完美分类，0.5 = 随机猜。

**Q: K-Means 有什么缺点？**
A: 1) 需要预先指定 K；2) 对初始化敏感；3) 只能发现球形簇；4) 对异常值敏感。

**Q: StratifiedKFold 和普通 KFold 的区别？**
A: Stratified 保证每个 fold 中正负样本比例和总体一致，防止某个 fold 全是负样本导致评估偏差。分类任务应该用 StratifiedKFold。

**Q: 为什么 PCA 对 KNN 有效但对 DT 没用？**
A: KNN 依赖距离，特征多时距离无意义（维度诅咒），PCA 降维后距离计算有意义。DT 本身在每个节点单独选特征，不受维度诅咒影响。

**Q: MLP 的 alpha 参数是什么？**
A: L2 正则化系数（权重衰减），惩罚大权重，防止过拟合。alpha 越大，正则化越强，模型越简单。
