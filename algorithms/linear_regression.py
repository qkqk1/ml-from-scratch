import numpy as np


class LinearRegressionGD:
    """
    Линейная регрессия через градиентный спуск с L2-регуляризацией.

    Параметры
    ----------
    lr : float
        Скорость обучения (learning rate).
    n_iter : int
        Количество итераций градиентного спуска.
    l2 : float
        Коэффициент L2-регуляризации (Ridge).
    """

    def __init__(self, lr=0.01, n_iter=1000, l2=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.w = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Обучить модель на данных X, y.

        Параметры
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        n, d = X.shape

        # Добавляем столбец единиц для bias
        X_b = np.hstack([X, np.ones((n, 1))])

        # Инициализируем веса нулями
        self.w = np.zeros(d + 1)
        self.loss_history = []

        for _ in range(self.n_iter):
            y_pred = X_b @ self.w

            # Градиент MSE + L2
            grad = (2 / n) * (X_b.T @ (y_pred - y)) + 2 * self.l2 * self.w

            self.w = self.w - self.lr * grad

            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

        return self

    def predict(self, X):
        """
        Предсказать значения для X.

        Параметры
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Возвращает
        ----------
        np.ndarray, shape (n_samples,)
        """
        X_b = np.hstack([X, np.ones((X.shape[0], 1))])
        return X_b @ self.w
