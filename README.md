# ml-from-scratch

Реализация классических ML-алгоритмов с нуля на Python + NumPy без использования sklearn.  
Каждый алгоритм верифицирован сравнением с реализацией sklearn, отклонение < 0.001.

## Алгоритмы

### Линейная регрессия [`algorithms/linear_regression.py`](algorithms/linear_regression.py)
- Градиентный спуск с L2-регуляризацией (Ridge)
- История сходимости loss
- Проверено на `make_regression` (200 объектов, 3 признака)
- Отклонение весов от `sklearn.linear_model.Ridge` < 0.001

### Логистическая регрессия [`algorithms/logistic_regression.py`](algorithms/logistic_regression.py)
- Градиентный спуск с функцией потерь Binary Cross-Entropy
- Методы `predict_proba` и `predict` (порог 0.5)
- Проверено на `load_breast_cancer`
- Accuracy совпадает с `sklearn.linear_model.LogisticRegression`

### kNN [`algorithms/knn.py`](algorithms/knn.py)
- Векторизованная матрица евклидовых расстояний через broadcasting
- Формула: `‖a − b‖² = ‖a‖² + ‖b‖² − 2·aᵀb`
- Проверено на `load_wine`
- Accuracy совпадает с `sklearn.neighbors.KNeighborsClassifier`

## Структура репозитория

```
ml-from-scratch/
├── README.md
├── algorithms/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   └── knn.py
├── notebooks/
│   ├── 01_linear_regression.ipynb
│   ├── 02_logistic_regression.ipynb
│   └── 03_knn.ipynb
└── kaggle/
    ├── titanic/
    └── house_prices/
```

## Стек

Python, NumPy, Matplotlib, scikit-learn (для верификации)
