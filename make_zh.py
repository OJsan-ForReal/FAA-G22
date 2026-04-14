import json, sys, copy
sys.stdout.reconfigure(encoding='utf-8')

with open('c:/Users/wusak/Downloads/FAA/faaPROJETO/FAA-G22/projeto.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb_zh = copy.deepcopy(nb)
cells = nb_zh['cells']

# [0] 标题
cells[0]['source'] = (
    '# 机器学习基础 — 项目\n\n'
    '**第 22 组**\n\n'
    '### 成员\n\n'
    '- **jiyi Li** (62244)\n\n'
    '- **oujie Wu** (62228)\n\n'
    '- **josé lourenço** (62817)\n'
)

# [1] 问题定义
cells[1]['source'] = (
    '## 1. 问题定义\n\n'
    '**根据行为变量预测期末成绩**：\n\n'
    '- `study_hours`（学习时间）\n\n'
    '- `gaming_hours`（游戏时间）\n\n'
    '- `sleep_hours`（睡眠时间）\n\n'
    '- `exercise_minutes`（运动时间）\n\n'
    '- ...\n'
)

# [2] 目标变量
cells[2]['source'] = (
    '## 目标变量（Targets）\n\n'
    '### 1. 回归场景（y）\n\n'
    '预测 **Exam Score**（考试成绩），连续值，范围在 **0 到 100** 之间。\n\n'
    '### 2. 分类场景（y）\n\n'
    '将 _Exam Score_ 转换为离散的可解释类别。根据数据分布与教育标准，定义如下：'
)

# [3] 分类表格
cells[3]['source'] = (
    '**多分类（成绩范围 0 到 65）：**\n\n'
    '|**类别**|**Exam Score 区间**|\n'
    '|---|---|\n'
    '|**不及格（Insuficiente）**|0 到 24|\n'
    '|**及格（Suficiente）**|25 到 39|\n'
    '|**良好（Bom）**|40 到 65|\n'
)

# [4] 利益相关者
cells[4]['source'] = (
    '## 利益相关者（Stakeholders）\n\n'
    '- 学生本人\n\n'
    '- 教育机构 / 学校\n\n'
    '- 学业顾问 / 辅导老师\n\n'
    '- 学生支持服务部门\n'
)

# [5] 读取文件标题
cells[5]['source'] = '### 读取文件，获取特征变量和目标变量'

# [6] 数据加载代码
cells[6]['source'] = (
    'import pandas as pd\n'
    'import numpy as np\n'
    'from sklearn import tree\n'
    'from sklearn.model_selection import KFold, train_test_split\n'
    'import io\n'
    'import matplotlib.pyplot as plt\n'
    'import matplotlib.image as mpimg\n'
    'import pydot\n'
    'from sklearn.model_selection import GridSearchCV\n'
    'from sklearn.model_selection import StratifiedKFold, cross_validate\n'
    'from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n'
    'from sklearn.preprocessing import LabelEncoder\n'
    'from sklearn.preprocessing import OrdinalEncoder\n'
    'from sklearn.tree import DecisionTreeClassifier\n'
    'from sklearn.tree import plot_tree\n'
    '\n\n'
    '# 读取学生记录文件\n'
    "df_student_records = pd.read_csv('student_records_full.csv')\n"
    'df_colunas = df_student_records.columns\n'
    '\n'
    "df_y = df_student_records['exam_score']\n"
    'df_y_numerico = df_y.copy()  # 回归用：连续值\n'
    '\n\n'
    'def categorizar_exame_score(score):\n'
    '    if score < 25:\n'
    "        return 'Insuficiente'  # 不及格\n"
    '    elif 25 <= score < 40:\n'
    "        return 'Suficiente'    # 及格\n"
    '    else:\n'
    "        return 'Bom'           # 良好\n"
    '\n'
    'df_y_categorico = df_y.apply(categorizar_exame_score)\n'
    '\n\n'
    '# 使用 OrdinalEncoder 将类别转换为有序数字\n'
    'categories = [["Insuficiente", "Suficiente", "Bom"]]\n'
    'oe = OrdinalEncoder(categories=categories)\n'
    'df_y_categorico_ordinais = oe.fit_transform(df_y_categorico.to_frame())\n'
    '\n'
    'print(df_y_categorico.value_counts())\n'
)

# [7] Pipeline 标题
cells[7]['source'] = (
    '## 2. 预处理流程（Pipeline）\n\n'
    '### 2.1 类别变量分析 — 编码方式选择依据'
)

# [8] 类别变量分析代码
cells[8]['source'] = (
    '# 分析类别变量的分布\n'
    'categoricas = ["gender", "academic_level", "internet_quality"]\n'
    '\n'
    'fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n'
    'for i, col in enumerate(categoricas):\n'
    '    counts = df_student_records[col].value_counts()\n'
    "    axes[i].bar(counts.index, counts.values, color='steelblue')\n"
    "    axes[i].set_title(f'分布: {col}')\n"
    '    axes[i].set_xlabel(col)\n'
    "    axes[i].set_ylabel('数量')\n"
    '    for j, v in enumerate(counts.values):\n'
    "        axes[i].text(j, v + 1, str(v), ha='center', fontsize=9)\n"
    '\n'
    'plt.tight_layout()\n'
    'plt.show()\n'
    '\n'
    '# 显示每个类别变量的唯一值\n'
    'for col in categoricas:\n'
    '    print(f"{col}: {sorted(df_student_records[col].unique())}")\n'
)

# [9] One-Hot 解释
cells[9]['source'] = (
    '**选择 One-Hot Encoding 的原因：**\n\n'
    '- `gender`（Male, Female, Other）：名义变量，类别之间没有自然顺序 → **One-Hot Encoding**\n'
    '- `academic_level`（High School, Bachelor, Master）：虽有隐含顺序，但各级别间的距离不均匀且无法量化 '
    '→ **One-Hot Encoding**，避免模型假设虚假的有序关系\n'
    '- `internet_quality`（Poor, Average, Good）：存在自然顺序，但选择 **One-Hot Encoding** '
    '是为了不强制施加任意数值尺度（0, 1, 2），以免扭曲线性模型'
)

# [10] 数值变量标题
cells[10]['source'] = '### 2.2 数值变量分析 — 标准化方式选择依据'

# [11] 数值变量分析代码
cells[11]['source'] = (
    'numericas = ["age", "study_hours", "self_study_hours", "online_classes_hours",\n'
    '             "social_media_hours", "gaming_hours", "sleep_hours",\n'
    '             "screen_time_hours", "exercise_minutes", "caffeine_intake_mg",\n'
    '             "mental_health_score"]\n'
    '\n'
    'df_num = df_student_records[numericas]\n'
    '\n'
    '# 绘制数值变量的直方图\n'
    "df_num.hist(figsize=(14, 10), bins=20, color='steelblue', edgecolor='white')\n"
    "plt.suptitle('数值变量分布', fontsize=14)\n"
    'plt.tight_layout()\n'
    'plt.show()\n'
    '\n'
    '# 计算偏度（Skewness）\n'
    'skewness = df_num.skew().round(3)\n'
    'print("各数值变量的偏度：")\n'
    'print(skewness)\n'
    'print("\\n规则：|skew| < 0.5 → 近似正态  |  |skew| > 1.0 → 严重偏斜")\n'
)

# [12] StandardScaler 解释
cells[12]['source'] = (
    '**选择 StandardScaler 的原因：**\n\n'
    '根据上方偏度分析，大多数数值变量的偏度较低至中等（|skew| < 1.0），接近正态分布。\n\n'
    '- **StandardScaler**（z = (x - μ) / σ）适用于近似正态分布，将数据中心化为均值0、标准差1\n'
    '- 对**线性回归**尤为重要，因为线性回归假设各特征处于可比的尺度\n'
    '- 对**决策树**，缩放不影响结果（决策树对尺度不敏感），但保持流程一致\n'
    '- `caffeine_intake_mg` 和 `exercise_minutes` 数值范围差异很大——不缩放会导致线性回归被这些变量主导'
)

# [13] 分类决策树标题
cells[13]['source'] = '### 分类场景 — 决策树（Decision Tree Classifier）'

# [14] 分类决策树代码
cells[14]['source'] = (
    'df_student_records.drop(columns=["student_id", "exam_score","productivity_score","burnout_level","focus_index"], inplace=True)\n'
    'df_categorico = df_student_records[["gender","academic_level","internet_quality"]]\n'
    'df_student_records_copy = df_student_records.copy()\n'
    '\n'
    '# One-Hot Encoding 处理类别变量\n'
    'df_X = pd.get_dummies(df_student_records_copy, columns=df_categorico.columns.values)\n'
    '\n'
    'X = df_X.values\n'
    'y = df_y_categorico_ordinais\n'
    '\n'
    '# 三路划分：学习集(80%) / 测试集(20%)，再将学习集分为训练集(60%) / 验证集(20%)\n'
    'X_learning, X_test, y_learning, y_test = train_test_split(\n'
    '    X, y, test_size=0.2, random_state=7, stratify=y)\n'
    '\n'
    'X_train, X_val, y_train, y_val = train_test_split(\n'
    '    X_learning, y_learning, test_size=0.25, random_state=7, stratify=y_learning)\n'
    '\n'
    '# 超参数搜索空间\n'
    'param_grid = {\n'
    '    "criterion": ["gini", "entropy"],\n'
    '    "max_depth": [1, 2, 3, 4, 5, 6, None],\n'
    '    "min_samples_split": [2, 5, 10, 20],\n'
    '    "min_samples_leaf": [1, 2, 5, 10],\n'
    '}\n'
    '\n'
    'grid = GridSearchCV(\n'
    '    estimator=tree.DecisionTreeClassifier(random_state=7),  # 固定随机种子确保可复现\n'
    '    param_grid=param_grid,\n'
    '    scoring="accuracy",  # 以准确率作为评估指标\n'
    '    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=7),  # 分层5折交叉验证\n'
    '    n_jobs=-1  # 使用所有CPU核心并行计算\n'
    ')\n'
    '\n'
    'grid.fit(X_learning, y_learning)  # 仅在学习集上训练和验证\n'
    '\n'
    'print("最优参数:", grid.best_params_)\n'
    'print("交叉验证最优准确率:", round(grid.best_score_, 3))\n'
    '\n'
    '# 使用最优参数的模型在测试集上评估\n'
    'final_model = grid.best_estimator_\n'
    'y_test_pred = final_model.predict(X_test)\n'
    'y_test_prob = final_model.predict_proba(X_test)[:, 1]\n'
    '\n'
    'final_metrics = pd.Series({\n'
    '    "Accuracy": accuracy_score(y_test, y_test_pred),\n'
    '    "Precision (macro)": precision_score(y_test, y_test_pred, average="macro"),\n'
    '    "Precision (weighted)": precision_score(y_test, y_test_pred, average="weighted"),\n'
    '    "Recall (macro)": recall_score(y_test, y_test_pred, average="macro"),\n'
    '    "Recall (weighted)": recall_score(y_test, y_test_pred, average="weighted"),\n'
    '    "F1-score (macro)": f1_score(y_test, y_test_pred, average="macro"),\n'
    '    "F1-score (weighted)": f1_score(y_test, y_test_pred, average="weighted"),\n'
    '})\n'
    '\n'
    'final_metrics.round(3)\n'
)

# [16] 回归决策树标题
cells[16]['source'] = '### 回归场景 — 决策树（Decision Tree Regressor）'

# [17] 回归决策树代码
cells[17]['source'] = (
    'import pandas as pd\n'
    'import numpy as np\n'
    'from sklearn import tree\n'
    'from sklearn.model_selection import KFold, train_test_split, GridSearchCV\n'
    'from sklearn.tree import DecisionTreeRegressor\n'
    'from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n'
    '\n'
    'y = df_y_numerico\n'
    'X = df_X.values\n'
    '\n'
    'X_learning, X_test, y_learning, y_test = train_test_split(\n'
    '    X, y, test_size=0.2, random_state=7\n'
    ')\n'
    '\n'
    'X_train, X_val, y_train, y_val = train_test_split(\n'
    '    X_learning, y_learning, test_size=0.25, random_state=7\n'
    ')\n'
    '\n'
    '# 超参数搜索空间\n'
    'param_grid = {\n'
    '    "criterion": ["squared_error", "absolute_error"],\n'
    '    "max_depth": [1, 2, 3, 4, 5, 6, None],\n'
    '    "min_samples_split": [2, 5, 10, 20],\n'
    '    "min_samples_leaf": [1, 2, 5, 10],\n'
    '}\n'
    '\n'
    'grid = GridSearchCV(\n'
    '    estimator=DecisionTreeRegressor(random_state=7),  # 回归决策树\n'
    '    param_grid=param_grid,\n'
    '    scoring="neg_root_mean_squared_error",  # 优化目标：最小化 RMSE\n'
    '    cv=KFold(n_splits=5, shuffle=True, random_state=7),  # 普通5折交叉验证\n'
    '    n_jobs=-1\n'
    ')\n'
    '\n'
    'grid.fit(X_learning, y_learning)  # 在学习集上训练和验证\n'
    '\n'
    'print("最优参数:", grid.best_params_)\n'
    '# GridSearchCV 将误差存为负数，取反还原\n'
    'print("交叉验证最优 RMSE:", round(-grid.best_score_, 3))\n'
    '\n'
    '# 在测试集上做最终评估\n'
    'final_model = grid.best_estimator_\n'
    'y_test_pred = final_model.predict(X_test)\n'
    '\n'
    '# 计算回归评估指标\n'
    'rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))\n'
    'mae = mean_absolute_error(y_test, y_test_pred)\n'
    'r2 = r2_score(y_test, y_test_pred)\n'
    '\n'
    'print("\\n--- 测试集评估指标 ---")\n'
    'print(f"RMSE（均方根误差）: {round(rmse, 3)}")\n'
    'print(f"MAE（平均绝对误差）: {round(mae, 3)}")\n'
    'print(f"R2 Score: {round(r2, 3)}")\n'
)

# [19] 线性回归标题
cells[19]['source'] = '### 线性回归（Linear Regression）'

# [20] 线性回归代码
cells[20]['source'] = (
    'from sklearn.linear_model import LinearRegression\n'
    'from sklearn.preprocessing import StandardScaler\n'
    'from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, max_error\n'
    'from scipy.stats import pearsonr\n'
    '\n'
    'y = df_y_numerico\n'
    'df_student_records_numerico = df_student_records.copy()\n'
    'df_student_records_numerico.drop(columns=["gender", "academic_level", "internet_quality"], inplace=True)\n'
    '\n'
    'df_xx = pd.get_dummies(df_student_records_copy, columns=df_categorico.columns.values, drop_first=True)\n'
    '\n'
    'X_train, X_test, y_train, y_test = train_test_split(\n'
    '    df_X, y, test_size=0.2, random_state=7\n'
    ')\n'
    '\n'
    '# 使用 StandardScaler：z = (x - u) / s，只对数值变量缩放\n'
    'scaler = StandardScaler()\n'
    '\n'
    '# 为避免修改原表，创建副本\n'
    'X_train_scaled = X_train.copy()\n'
    'X_test_scaled = X_test.copy()\n'
    '\n'
    '# 训练集用 fit_transform，测试集只能用 transform（防止数据泄漏）\n'
    'X_train_scaled[df_student_records_numerico.columns] = scaler.fit_transform(X_train[df_student_records_numerico.columns])\n'
    'X_test_scaled[df_student_records_numerico.columns] = scaler.transform(X_test[df_student_records_numerico.columns])\n'
    '\n'
    'reg = LinearRegression().fit(X_train_scaled, y_train)\n'
    'print("训练集 R2 得分: ", reg.score(X_train_scaled, y_train))\n'
    'print("截距（alpha）: ", reg.intercept_)\n'
    'print("各特征系数：")\n'
    'for i, beta in enumerate(reg.coef_):\n'
    '    print("\\t B%d -> %9.3f"% (i+1, beta))\n'
    '\n\n'
    'def printRegStatistics(truth, preds):\n'
    '    print("RMSE（均方根误差）: ", root_mean_squared_error(truth, preds))  # 越小越好\n'
    '    print("MAE（平均绝对误差）: ", mean_absolute_error(truth, preds))     # 越小越好\n'
    '    print("R2（决定系数）: ", r2_score(truth, preds))                     # 越接近1越好\n'
    '    corr, pval = pearsonr(truth, preds)\n'
    '    print("皮尔逊相关系数: %6.4f (p-value=%e)\\n"%(corr, pval))\n'
    '    print("最大误差: ", max_error(truth, preds))\n'
    '\n\n'
    'preds = reg.predict(X_test_scaled)\n'
    'printRegStatistics(y_test, preds)\n'
    '\n'
    'plt.figure(figsize=(7,7))\n'
    'plt.scatter(preds, y_test)\n'
    'plt.grid()\n'
    '# 绘制完美预测参考线（红色虚线）\n'
    "plt.plot([0, 100], [0, 100], c='r', linestyle='--', linewidth=2)\n"
    'plt.xlim(0, 100)\n'
    'plt.ylim(0, 100)\n'
    'plt.xlabel("预测值")\n'
    'plt.ylabel("真实值")\n'
    'plt.show()\n'
)

with open('c:/Users/wusak/Downloads/FAA/faaPROJETO/FAA-G22/projeto_zh.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_zh, f, ensure_ascii=False, indent=1)

print('Done! projeto_zh.ipynb 已生成。')
