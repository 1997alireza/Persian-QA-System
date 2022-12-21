from sklearn import svm
import numpy as np
import pickle
from src.modelling.classifiers.abstract_classifier_model import ClassifierModel


class SVM_Model(ClassifierModel):

    def __init__(self):
        self.clf = svm.SVC(gamma='scale', probability=True)

    def train(self, features, labels_id):
        self.clf.fit(features, labels_id)

    def get_probabilities(self, feature):
        return self.clf.predict_proba(np.reshape(feature, (1, -1)))[0]

    def predict_one(self, feature):
        return int(self.clf.predict(np.reshape(feature, (1, -1)))[0])  # (1, -1) -> one sample, several feature items

    def predict(self, features):
        return self.clf.predict(features)

    def save(self, address):
        model_file = open(address, 'wb+')
        pickle.dump(self, model_file)
        model_file.close()

    @staticmethod
    def load(address):
        model_file = open(address, 'rb')
        model = pickle.load(model_file)
        model_file.close()
        return model
