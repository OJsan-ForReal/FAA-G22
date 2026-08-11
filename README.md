# Student Grade Prediction — Behavioral Data

Academic project for **Fundamentos de Aprendizagem Automática (FAA)**, FCUL —
Group 22.

Predicts exam performance from student behavioral data (sleep hours, screen
time, study/leisure balance, mental health score, and similar variables),
approached both as a **classification** problem (pass/fail) and a
**regression** problem (predicted score), with a full pipeline from
preprocessing through feature selection, model comparison, and
interpretability analysis.

**Team**: Jiyi Li, Ojie (Oujie Wu), José Lourenço

## Notebooks

- [`projeto_G22_Classificacao.ipynb`](projeto_G22_Classificacao.ipynb) —
  binary classification (Insuficiente / Suficiente)
- [`projeto_G22_Regressao.ipynb`](projeto_G22_Regressao.ipynb) — regression
  on the raw exam score

Notebook content and course deliverable (`FAA2526-projeto1.pdf`) are in
Portuguese.

## Approach

**Preprocessing** — missing-value handling (two dataset variants are
included: `student_records_full.csv` and `student_records_missing.csv`),
one-hot encoding for nominal categoricals, `StandardScaler` for
distance-based models.

**Class imbalance** — the pass/fail split is ~71/29, so accuracy alone would
reward a model that just predicts the majority class. **F1-score** is used
as the primary evaluation metric for classification instead.

**Feature selection** — five techniques compared systematically across five
model families: Variance Threshold, Correlation Filter, Mutual Information,
Sequential Feature Selection (SFS), and PCA.

**Models** — Decision Tree, KNN, Gaussian Naive Bayes, Random Forest, and
MLP for classification; Decision Tree, Linear Regression, KNN, Random
Forest, and MLP for regression. Each combination is tuned with
`GridSearchCV` + cross-validation.

## Results (classification)

| Model | Feature selection | F1-score |
|---|---|---|
| KNN | baseline (17 features) | 0.5313 |
| KNN | Sequential Feature Selection (8 features) | **0.6286** |
| Gaussian NB | — | ~0.66 |
| Decision Tree | — | 0.654 |

SFS gave the largest single improvement (+18.3% F1 over the KNN baseline)
by cutting noisy features rather than adding data. Across the full
comparison: MLP had the strongest raw predictive performance but is slower
to train and more hyperparameter-sensitive; Gaussian NB is fast, needs no
scaling, and gets close (~0.66) with far less tuning; Decision Tree trades
a bit of performance for fully interpretable rules.

## Unsupervised — behavioral clustering

K-Means on the behavioral features, K chosen via the elbow method and
confirmed with silhouette score (K=4: 0.07138 vs K=2: 0.06826 — inertia
drop also flattens noticeably past K=4). Four student personas emerged:

- **Self-disciplined & active** — older, more study time, more exercise,
  less sleep, higher psychological stress
- **Focused on academic life** — younger, heavy autonomous study time,
  low social/leisure activity, more postgraduate students
- **Socially active** — heavy online classes, high social media/gaming
  time, higher caffeine intake
- *(a fourth persona rounds out the K=4 clustering)*

## Interpretability & error analysis

SHAP (SHapley Additive exPlanations) applied to the black-box models
(Random Forest / MLP) and compared against the interpretable Decision Tree.
Both notebooks go beyond aggregate metrics into **manual case-by-case error
inspection**:

- Classification: false positives/negatives are pulled individually and
  explained via SHAP — e.g. a student with high `study_hours` but poor
  `internet_quality` gets over-predicted as passing.
- Regression: residuals are split into over-estimation vs under-estimation
  patterns — e.g. high `study_hours` with low `mental_health_score` tends
  to be over-predicted; high `gaming_hours`/`social_media_hours` tends to
  be under-predicted relative to actual outcome.

## Files

- `projeto_G22_Classificacao.ipynb`, `projeto_G22_Regressao.ipynb` — the two notebooks
- `student_records_full.csv`, `student_records_missing.csv` — dataset (complete / with missing values)
- `FAA2526-projeto1.pdf` — original course assignment brief
