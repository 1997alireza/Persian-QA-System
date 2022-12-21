from abc import abstractmethod


class ClassifierModel:
    @abstractmethod
    def train(self, features, labels_id):
        pass

    @abstractmethod
    def get_probabilities(self, feature):
        # returns a numpy.ndarray
        pass

    @abstractmethod
    def predict_one(self, feature):
        pass

    @abstractmethod
    def predict(self, features):
        pass

    @abstractmethod
    def save(self, address):
        pass

    @staticmethod
    def load(address):
        pass
