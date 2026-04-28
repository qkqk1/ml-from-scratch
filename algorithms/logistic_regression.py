import numpy as np


class LogisticRegressionGD:
    """
    Логистическая регрессия через градиентный спуск (Binary Cross-Entropy).

    Параметры
    ----------
    lr : float
        Скорость обучения (learning rate).
    n_iter : int
        Количество итераций градиентного спуска.
    """

    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w_ = None
        self.loss_history_ = []

    def sigmoid(self, z):
        """
        Сигмоида: σ(z) = 1 / (1 + e^{-z}).

        Параметры
        ----------
        z : np.ndarray

        Возвращает
        ----------
        np.ndarray — значения от 0 до 1
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Обучить модель на данных X, y.

        Параметры
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — бинарные метки 0/1
        """
        n = X.shape[0]

        # Добавляем столбец единиц для bias
        X_b = np.hstack([X, np.ones((n, 1))])

        # Инициализируем веса нулями
        self.w_ = np.zeros(X_b.shape[1])
        self.loss_history_ = []

        for _ in range(self.n_iter):
            # Предсказание вероятностей
            y_pred = self.predict_proba(X)

            # Градиент BCE: X^T (ŷ - y) / n
            grad = (1 / n) * (X_b.T @ (y_pred - y))

            # Обновить веса
            self.w_ = self.w_ - self.lr * grad

            # Сохранить loss
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            self.loss_history_.append(loss)

        return self

    def predict_proba(self, X):
        """
        Предсказать вероятности класса 1.

        Параметры
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Возвращает
        ----------
        np.ndarray, shape (n_samples,) — вероятности от 0 до 1
        """
        X_b = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.sigmoid(X_b @ self.w_)

    def predict(self, X):
        """
        Предсказать классы (порог 0.5).

        Параметры
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Возвращает
        ----------
        np.ndarray, shape (n_samples,) — предсказанные классы 0/1
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
