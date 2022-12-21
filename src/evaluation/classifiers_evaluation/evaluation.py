import paths
from src.processing.feature_extractor import extract_labels_and_features, extract_splitted_ready_sets
from src.evaluation.classifiers_evaluation.helper import k_fold_test, test_model_on_sets
from src.processing.classifier_learner import get_model_and_feature_class


def evaluation(accuracy_test_type,
               model_specification='CNN',
               question_relation_file_adr=paths.dataset+'rel_q_expanded.csv'):
    """
    :param accuracy_test_type: can be 'splitted-ready-dataset' (with several options which come after it with spaces) or 'k-fold'
    """

    model_class, sentence_feature_style = get_model_and_feature_class(model_specification)

    assert accuracy_test_type == 'k-fold' or \
           accuracy_test_type.split()[0] == 'splitted-ready-dataset', 'Wrong accuracy test type: ' + accuracy_test_type

    if accuracy_test_type == 'k-fold':
        vocab, labels_set, labels_id, features = extract_labels_and_features(question_relation_file_adr,
                                                                             sentence_feature_style)
        correctness_ratio, precision, recall, f1_score = \
            k_fold_test(vocab, labels_set, features, labels_id, model_class=model_class, k=5)
        print('k-fold test: accuracy={}, precision={}, recall={}, f1_score={}'.format(correctness_ratio,
                                                                                      precision, recall,
                                                                                      f1_score))
    else:  # accuracy_test_type.split()[0] == 'splitted-ready-dataset'
        options = accuracy_test_type.split()[1:]
        if 'distinct' in options:
            datasets_dir = paths.dataset + 'splitted_distinct_sets/'
        else:
            datasets_dir = paths.dataset + 'splitted_sets/'
        if 'unidirectional' in options:
            datasets_dir = datasets_dir + 'unidirectional/'

        vocab, labels_set, train_labels_id, train_features, _, _, test_labels_id, test_features, test_sentences = \
            extract_splitted_ready_sets(datasets_dir, sentence_feature_style, parse_validation_file=False)

        print('datasets are parsed')

        correctness_ratio, precision, recall, f1_score = \
            test_model_on_sets(
                vocab, labels_set, train_features, train_labels_id, test_features, test_labels_id, test_sentences,
                model_class=model_class)
        print('test on ready sets: accuracy={}, precision={}, recall={}, f1_score={}'.format(
            correctness_ratio, precision, recall, f1_score))


if __name__ == '__main__':
    evaluation('splitted-ready-dataset distinct')
