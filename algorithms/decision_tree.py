import numpy as np


class Node:
    """
    Узел дерева решений.

    Параметры
    ----------
    feature_idx : int или None
        Индекс признака для разбиения (None у листового узла).
    threshold : float или None
        Порог разбиения (None у листового узла).
    left : Node или None
        Левое поддерево (объекты >= threshold).
    right : Node или None
        Правое поддерево (объекты < threshold).
    value : int или None
        Предсказание листового узла (мода классов). None у внутренних узлов.
    """

    def __init__(self, feature_idx=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def gini(y):
    """
    Критерий Джини для массива меток.

    Gini = 1 - sum(p_k^2)

    Параметры
    ----------
    y : np.ndarray, shape (n,) — метки классов

    Возвращает
    ----------
    float — значение критерия Джини
    """
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - (probs ** 2).sum()


def best_split(X, y):
    """
    Найти лучшее разбиение по минимуму взвешенного критерия Джини.

    Параметры
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,) — метки классов

    Возвращает
    ----------
    best_feature : int — индекс признака
    best_threshold : float — порог разбиения
    """
    best_feature, best_threshold, best_gini = None, None, np.inf

    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])

        for threshold in thresholds:
            left_y = y[X[:, feature_idx] >= threshold]
            right_y = y[X[:, feature_idx] < threshold]

            # Взвешенный джини: n_L/n * Gini(L) + n_R/n * Gini(R)
            gini_split = (len(left_y) * gini(left_y) +
                          len(right_y) * gini(right_y)) / len(y)

            if gini_split < best_gini:
                best_gini = gini_split
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold


class DecisionTree:
    """
    Дерево решений для классификации (критерий Джини, CART).

    Параметры
    ----------
    max_depth : int
        Максимальная глубина дерева.
    min_samples : int
        Минимальное число объектов для разбиения узла.
    """

    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def fit(self, X, y):
        """
        Построить дерево по обучающей выборке.

        Параметры
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — метки классов
        """
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        # Условия останова — вернуть листовой узел
        if len(y) < self.min_samples or depth >= self.max_depth or gini(y) == 0:
            return Node(value=np.bincount(y).argmax())

        feature, threshold = best_split(X, y)

        left_mask = X[:, feature] >= threshold
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)

        return Node(feature, threshold, left, right)

    def predict(self, X):
        """
        Предсказать классы для объектов X.

        Параметры
        ----------
        X : np.ndarray, shape (m_samples, n_features)

        Возвращает
        ----------
        np.ndarray, shape (m_samples,) — предсказанные классы
        """
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] >= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
