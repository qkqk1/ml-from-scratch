import numpy as np


class KNN:
    """
    k ближайших соседей для классификации (KNN).

    Параметры
    ----------
    k : int
        Количество ближайших соседей.
    """

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Запомнить обучающую выборку.

        Параметры
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — метки классов
        """
        self.X_train = X
        self.y_train = y
        return self

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
        # Матрица расстояний (m, n)
        distances = self._euclidean_distances(X)

        # Индексы k ближайших соседей для каждого объекта
        k_nearest = np.argsort(distances, axis=1)[:, :self.k]

        # Метки k ближайших соседей
        k_labels = self.y_train[k_nearest]

        # Для каждой строки — самая частая метка (мода)
        predictions = np.apply_along_axis(
            lambda row: np.argmax(np.bincount(row)),
            axis=1,
            arr=k_labels
        )
        return predictions

    def _euclidean_distances(self, X):
        """
        Вычислить матрицу евклидовых расстояний между X и X_train.

        Использует формулу: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a^T * b

        Параметры
        ----------
        X : np.ndarray, shape (m, d) — тестовые объекты

        Возвращает
        ----------
        np.ndarray, shape (m, n) — матрица расстояний
        """
        X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
        X_train_sq = np.sum(self.X_train ** 2, axis=1).reshape(1, -1)
        dot = X @ self.X_train.T

        return np.sqrt(X_sq + X_train_sq - 2 * dot)
