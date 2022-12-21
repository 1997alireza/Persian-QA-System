from src.preparation.data_cleaner import final_dataset_generator
from src.processing.feature_extractor import generate_idf_file
from src.processing.classifier_learner import get_trained_rel_model
from src.processing.answer_generator import get_answer_generator


def generate_final_dataset(rel_question_source, append=False):
    """
    to generate the final dataset which is usable for models to train

    @param rel_question_source: a file contains the pairs of relations and question templates
    @param append: append new data to the last ones. if it's false, last dataset will be erased!
    """
    final_dataset_generator(rel_question_source, append)


def train():
    """
    train the classification model

    """
    # if SVM model with tf-idf or weighted word2vec embedding styles is going to train,
    # calling `generate_idf_file` before training would help this process
    # it's not usable now since we are using the CNN model
    # generate_idf_file()

    get_trained_rel_model(force_train_models=True)


def run(multiple_relations=False):
    """

    @param multiple_relations: if it's true, all of the relations with high probabilities is checked instead of just the best one

    """
    answer_generator = get_answer_generator()
    while True:
        input_sentence = input('Enter a sentence:')
        if input_sentence == 'q':
            exit()
        info = answer_generator(input_sentence, multiple_relations=multiple_relations)
        print(info, '\n')
