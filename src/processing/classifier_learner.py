from src.modelling.classifiers.svm import SVM_Model
from src.modelling.classifiers.cnn.cnn import CNN_Model
import paths
import os
import numpy as np
from src.processing.feature_extractor import feature_extractor, extract_labels_and_features, get_vocab_and_labels_set, \
    save_vocab_and_labels_set


def get_trained_rel_model(model_specification='CNN',

                          force_train_models=False,
                          question_relation_file_adr=paths.dataset+'rel_q_expanded.csv',
                          saving_directory=paths.saved_models
                          ):
    """
    attention: if you want to "train" a model that uses "tf-idf" or "weighted word2vec" embedding styles, you may
    rebuild the idf file using `generate_idf_file` function before calling this function. it helps the models to train
    and works more accurate.

    :param model_specification: could be 'CNN' or 'SVM {style}'. {style} is the name of the sentence feature style for SVM classification
    :param force_train_models: set true to not load the model from the file
    :param question_relation_file_adr: the main dataset which is used to train models
    :param saving_directory: saving models directory
    :return: it returns a function (detector) which takes a sentence and predicts its relation.
    """

    if force_train_models and not os.path.isfile(question_relation_file_adr):
        raise Exception('Dataset file is not find to train models')

    model_class, sentence_feature_style = get_model_and_feature_class(model_specification)

    if saving_directory[-1] != '/':
        saving_directory = saving_directory + '/'

    model_file_adr = \
        saving_directory + 'classifiers/' + model_class.__name__ + '/' + sentence_feature_style + ' style' + '.model'

    if os.path.isfile(model_file_adr) and not force_train_models:
        vocab, labels_set = get_vocab_and_labels_set()
        model = model_class.load(model_file_adr)
    else:
        vocab, labels_set, labels_id, features = extract_labels_and_features(question_relation_file_adr,
                                                                             sentence_feature_style)
        save_vocab_and_labels_set(vocab, labels_set)

        model = model_class()
        if model_class == CNN_Model:
            model.initialize(vocab, len(labels_set))

        model.train(features, labels_id)
        model.save(model_file_adr)

    def detector(input_sentence, multiple_relations=True):
        """
        :param multiple_relations: set it true when you want to get multiple answers, which are sorted based on their probabilities

        """
        if multiple_relations:
            probabilities = model.get_probabilities(
                feature_extractor(input_sentence, vocab, sentence_feature_style))
            sorted_indexes = np.argsort(probabilities)[::-1]
            sorted_rel_probs = []
            for i in sorted_indexes:
                sorted_rel_probs.append({
                    'rel': labels_set[i],
                    'prob': np.float64(probabilities[i])  # np.float64: to be JSON serializable for web interface
                })

            return sorted_rel_probs
        else:
            predicted_relation_id = model.predict_one(
                feature_extractor(input_sentence, vocab, sentence_feature_style))
            return labels_set[predicted_relation_id]
    return detector


def get_model_and_feature_class(model_specification):
    """
    :param model_specification: could be 'CNN' or 'SVM {style}'. {style} is the name of the sentence feature style for SVM classification
    """
    classification_model = model_specification.split(' ')[0].lower()

    if classification_model == 'svm':
        try:
            sentence_feature_style = model_specification.split(' ')[1].lower()
        except IndexError:
            sentence_feature_style = 'word2vec'

        assert sentence_feature_style == 'bow' or sentence_feature_style == 'tf-idf' or sentence_feature_style == 'word2vec', \
            'Wrong word feature style' + '\n' + '    Requested style: ' + '\n' + sentence_feature_style

        return SVM_Model, sentence_feature_style

    elif classification_model == 'cnn':
        return CNN_Model, 'word_vocab_ids_increased_by_one_for_cnn'


    raise Exception('Unsupported classification model')


if __name__ == '__main__':
    label_detector = get_trained_rel_model(model_specification='CNN')

    print('The model is ready')
    while True:
        input_sentence = input('Enter a sentence:')
        if input_sentence == 'q':
            exit()
        print(label_detector(input_sentence), '\n')

