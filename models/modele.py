from abc import ABC, abstractmethod


class Modele(ABC):

    def __init__(self):
        self.w = None  # doit-être définit pas la méthode entraînement

    @abstractmethod
    def entrainement(self, x_train, t_train):
        pass

    @abstractmethod
    def prediction(self, x):
        pass

    @staticmethod
    @abstractmethod
    def erreur(t, prediction):
        pass
