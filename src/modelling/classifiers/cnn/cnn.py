from src.modelling.classifiers.abstract_classifier_model import ClassifierModel
from src.modelling.classifiers.cnn.embedding_helper import read_glove_vectors
from src.modelling.classifiers.cnn.model_provider import get_model
import numpy as np
from keras.models import load_model
import paths

MAX_SEQUENCE_LEN = 32


class CNN_Model(ClassifierModel):
    def __init__(self, model=None):
        self.model = model
        self.labels_count = None

    def initialize(self, vocab, labels_count, embedding_file_path=paths.word2vec):
        """need to be called before the train process"""
        embeddings_index, embedding_dim = read_glove_vectors(embedding_file_path)

        vocab_size = len(vocab)

        embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
        # embedding_matrix[0] remains a zero vector, it's assigned to the padding value (0),
        # due to the `get_word_vocab_id_vector` function
        # and because of that, each vocab id increases by one

        for i in range(vocab_size):
            word = vocab[i]
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i+1] = embedding_vector

        self.labels_count = labels_count
        self.model = get_model(embedding_matrix, vocab_size, embedding_dim, labels_count, MAX_SEQUENCE_LEN)

    def train(self, features, labels_id):
        features_np = np.array(features, dtype='float32')
        labels_id_np = np.zeros([len(labels_id), self.labels_count], dtype='float32')

        for i in range(len(labels_id)):
            correct_id = labels_id[i]
            labels_id_np[i][correct_id] = 1.
        self.model.fit(features_np, labels_id_np, epochs=10, batch_size=32)

    def get_probabilities(self, feature):
        return self.model.predict(np.reshape(feature, (1, -1)))[0]

    def predict_one(self, feature):
        labels_probability = self.get_probabilities(feature)
        return np.argmax(labels_probability)

    def predict(self, features):
        prediction_list = []
        features = np.array(features)
        probabilities = self.model.predict(features)
        for probs in probabilities:
            prediction_list.append(np.argmax(probs))
        return prediction_list

    def save(self, address):
        self.model.save(address)

    @staticmethod
    def load(address):
        """attention: after this, self.labels_count is still None which is required for train process"""
        model = load_model(address)
        model._make_predict_function()
        return CNN_Model(model)
