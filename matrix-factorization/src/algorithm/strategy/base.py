from abc import ABC, abstractmethod


class FitStrategy(ABC):
    W = None
    U = None
    K = None
    reg = 0.0

    @abstractmethod
    def init_params(self, M, N, K, mu=None, reg=0.0):
        pass

    @abstractmethod
    def step(self, items_per_user, users_per_item):
        pass

    @abstractmethod
    def predict(self, row):
        pass
