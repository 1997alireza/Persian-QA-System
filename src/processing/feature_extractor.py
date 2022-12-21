from math import log
import numpy as np
import pickle
import pandas as pd
import os
from src.utils.tools import extract_dataset_sentences
import paths


class DataKeeper:
    vocab_idf_list = None
    vocab_idf_mean = 0  # is used for the words which are not in the vocab idf list
    word2vec_wiki = None


def make_vocab(sentences):
    """"
    the order should be fix on each run
    param sentences
    return: vocabulary
    """
    vocab = []
    for s in sentences:
        for w in s.split():
            if w not in vocab:
                vocab.append(w)
    return vocab


def feature_extractor_batch(sentences, vocab, sentence_feature_style):
    assert sentence_feature_style == 'bow' or sentence_feature_style == 'tf-idf' or sentence_feature_style == 'word2vec' \
           or sentence_feature_style == 'word_vocab_ids_increased_by_one_for_cnn'

    return [feature_extractor(s, vocab, sentence_feature_style)
            for s in sentences if len(s.replace(' ', '')) > 0]


def feature_extractor(sentence, vocab, sentence_feature_style):
    s_splitted = sentence.split()
    if sentence_feature_style == 'bow':
        return get_bow(s_splitted, vocab)
    if sentence_feature_style == 'tf-idf':
        return get_tfidf(s_splitted, vocab)
    if sentence_feature_style == 'word2vec':
        return get_word2vec(s_splitted, vocab, weighted=True)
    if sentence_feature_style == 'word_vocab_ids_increased_by_one_for_cnn':
        return get_word_vocab_id_vector_for_cnn(s_splitted, vocab)

    raise ValueError('Wrong word feature style' + '\n' + '    Requested style: ' + '\n' + sentence_feature_style)


def get_bow(s_splitted, vocab):
    s_bow = []
    for v in vocab:
        s_bow.append(s_splitted.count(v))
    return s_bow


def get_tfidf(s_splitted, vocab):
    vocab_idf_list, vocab_idf_mean = get_idf_list(vocab)
    s_tfidf = []
    for v_id, v in enumerate(vocab):
        try:
            idf = vocab_idf_list[v]
        except:
            idf = vocab_idf_mean
        tf = s_splitted.count(v)
        s_tfidf.append(tf * idf)
    return s_tfidf


def get_word2vec(s_splitted, vocab=None, weighted=True):
    vector_length = word2vec(get_size=True)

    features_sum = np.array([0] * vector_length, dtype=np.float64)
    total_weight = 0.

    if weighted:
        vocab_idf_list, vocab_idf_mean = get_idf_list(vocab)
        for w in s_splitted:
            v = word2vec(w)
            if v is not None:
                try:
                    w_idf = vocab_idf_list[w]
                except:
                    w_idf = vocab_idf_mean

                features_sum = features_sum + v * w_idf
                total_weight += w_idf

    else:
        for w in s_splitted:
            v = word2vec(w)
            if v is not None:
                features_sum = features_sum + v
                total_weight += 1

    if total_weight == 0:  # non of the input words are not in the word2vec or idf list
        return list(features_sum)

    return list(features_sum / total_weight)


def get_word_vocab_id_vector_for_cnn(s_splitted, vocab):
    """
    Sequences that are shorter than `word_vector_size`
    are padded at the end.
    Sequences longer than `word_vector_size` are truncated.

    increased_by_one: increase each vocab id, so non of the words'id is zero, because zero is reserved for padding value
    """
    from src.modelling.classifiers.cnn.cnn import MAX_SEQUENCE_LEN
    sequence_length = MAX_SEQUENCE_LEN

    padding_value = 0  # cannot be changed
    word_id_vector = np.array([padding_value] * sequence_length, dtype=np.float64)

    for i in range(min(len(s_splitted), sequence_length)):
        is_found = False
        for v_id, v in enumerate(vocab):
            if s_splitted[i] == v:
                is_found = True
                word_id_vector[i] = v_id + 1
                continue
        if not is_found:
            word_id_vector[i] = padding_value

    return word_id_vector


def get_idf_list(vocab=None, force_rebuild_idf_file=False):
    if DataKeeper.vocab_idf_list is None:  # multiple use in a single run

        if force_rebuild_idf_file and vocab is None:
            raise Exception('You should provide a vocab list to build the idf file.')
        if not os.path.isfile(paths.vocab_idf) and vocab is None:
            raise Exception('Before using the models, train the idf file using `generate_idf_file` function.')

        if os.path.isfile(paths.vocab_idf) and not force_rebuild_idf_file:
            with open(paths.vocab_idf, 'rb') as f:
                vocab_idf_list = pickle.load(f)
        else:
            print('start making vocab idf list file')
            tfidf_docs = []
            with open(paths.dataset + 'tf-idf_train_set.txt', 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if idx % 2 == 0:  # even lines are empty in 'tf-idf_train_set.txt' file
                        tfidf_docs.append(line)

            vocab_idf_list = {}
            found_words = {}  # to check found words in each doc
            tfidf_docs_len = len(tfidf_docs)
            for v in vocab:
                vocab_idf_list[v] = 1  # to avoid to be zero so division by zero would not happen
                found_words[v] = False

            for doc_id, doc in enumerate(tfidf_docs):
                found_words = dict.fromkeys(found_words, False)
                for word in doc.split():
                    try:
                        found_words[word] |= True  # use |= operation to prevent creating a new key
                    except KeyError:
                        continue

                for v in found_words:
                    if found_words[v]:
                        vocab_idf_list[v] += 1

            for v in vocab:
                idf = log(tfidf_docs_len / vocab_idf_list[v])  # less repeat more value
                vocab_idf_list[v] = idf

            # to save IDFs of the words to avoid computing again in each run
            with open(paths.vocab_idf, 'wb+') as f:
                pickle.dump(vocab_idf_list, f)
            print('vocab idf list saved')

        idf_values = list(vocab_idf_list.values())
        idf_mean = sum(idf_values) / len(idf_values)

        DataKeeper.vocab_idf_list = vocab_idf_list
        DataKeeper.vocab_idf_mean = idf_mean

    return DataKeeper.vocab_idf_list, DataKeeper.vocab_idf_mean


def generate_idf_file(question_relation_file_adr=paths.dataset+'rel_q_expanded.csv'):
    """
    to generate the idf file
    if you want to rebuild the idf file, use this function
    """
    _, sentences = extract_dataset_sentences(question_relation_file_adr)
    vocab = make_vocab(sentences)
    return get_idf_list(vocab, force_rebuild_idf_file=True)


def word2vec(word=None, get_size=False):
    """
    if `get_size` is true, it just returns the length of the vectors, else it returns the vector of the word
    """
    if DataKeeper.word2vec_wiki is None:
        w_list = []
        coefs_list = []
        with open(paths.word2vec, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:-1]
            for line in lines:
                splitted = line.split(' ')
                coefs = [float(f) for f in splitted[1:]]
                w = splitted[0]
                w_list.append(w)
                coefs_list.append(coefs)
        w2v_wiki = (w_list, np.array(coefs_list))
        DataKeeper.word2vec_wiki = w2v_wiki
    else:
        w2v_wiki = DataKeeper.word2vec_wiki

    if get_size:
        return len(w2v_wiki[1][0])

    try:
        idx = w2v_wiki[0].index(word)
        return w2v_wiki[1][idx]
    except:
        return None  # the word is not in the train set


def extract_labels_and_features(question_relation_file_adr, sentence_feature_style):
    labels, sentences = extract_dataset_sentences(question_relation_file_adr)
    vocab = make_vocab(sentences)

    features = feature_extractor_batch(sentences, vocab, sentence_feature_style)

    # the order should be fix on each run
    labels_set = []
    for l in labels:
        if l not in labels_set:
            labels_set.append(l)

    labels_id = []
    for l in labels:
        labels_id.append(labels_set.index(l))

    return vocab, labels_set, labels_id, features


def extract_vocab_and_labels_set(question_relation_file_adr):
    labels, sentences = extract_dataset_sentences(question_relation_file_adr)
    vocab = make_vocab(sentences)

    # the order should be fix on each run
    labels_set = []
    for l in labels:
        if l not in labels_set:
            labels_set.append(l)

    return vocab, labels_set


def get_vocab_and_labels_set():
    if not (os.path.exists(paths.vocab) and os.path.exists(paths.labels)):
        raise Exception('Vocab or labels files are not found. Train again using your dataset.')

    try:
        with open(paths.vocab, 'rb') as f:
            vocab = pickle.load(f)
        with open(paths.labels, 'rb') as f:
            labels_set = pickle.load(f)
    except Exception:
        raise Exception('Vocab or labels files are corrupted. Train again using your dataset.')


    return vocab, labels_set


def save_vocab_and_labels_set(vocab, labels_set):
    with open(paths.vocab, 'wb') as f:
        pickle.dump(vocab, f)
    with open(paths.labels, 'wb') as f:
        pickle.dump(labels_set, f)


def extract_splitted_ready_sets(datasets_directory, sentence_feature_style, parse_validation_file=True):
    if datasets_directory[-1] != '/':
        datasets_directory = datasets_directory + '/'

    train_set = pd.read_csv(datasets_directory + 'train.csv', header=None)
    test_set = pd.read_csv(datasets_directory + 'test.csv', header=None)

    train_labels = train_set.iloc[:, 0].values.tolist()
    train_sentences = train_set.iloc[:, 1].values.tolist()

    test_labels = test_set.iloc[:, 0].values.tolist()
    test_sentences = test_set.iloc[:, 1].values.tolist()

    if parse_validation_file:
        validation_set = pd.read_csv(datasets_directory + 'validation.csv', header=None)
        validation_labels = validation_set.iloc[:, 0].values.tolist()
        validation_sentences = validation_set.iloc[:, 1].values.tolist()
    else:
        validation_labels = []
        validation_sentences = []


    if len(train_set.iloc[0]) > 2:
        train_sentences = train_sentences + train_set.iloc[:, 3].values.tolist()
        if parse_validation_file:
            validation_sentences = validation_sentences + validation_set.iloc[:, 3].values.tolist()
        test_sentences = test_sentences + test_set.iloc[:, 3].values.tolist()

        train_labels *= 2
        validation_labels *= 2
        test_labels *= 2

    vocab = make_vocab(train_sentences + validation_sentences + test_sentences)


    train_features = feature_extractor_batch(train_sentences, vocab, sentence_feature_style)

    test_features = feature_extractor_batch(test_sentences, vocab, sentence_feature_style)

    if parse_validation_file:
        validation_features = feature_extractor_batch(validation_sentences, vocab, sentence_feature_style)
    else:
        validation_features = None


    # make set of all labels
    labels_set = []
    for l in train_labels + validation_labels + test_labels:
        if l not in labels_set:
            labels_set.append(l)

    train_labels_id = []
    for l in train_labels:
        train_labels_id.append(labels_set.index(l))

    test_labels_id = []
    for l in test_labels:
        test_labels_id.append(labels_set.index(l))

    if parse_validation_file:
        validation_labels_id = []
        for l in validation_labels:
            validation_labels_id.append(labels_set.index(l))
    else:
        validation_labels_id = None


    return vocab, labels_set, train_labels_id, train_features, validation_labels_id, validation_features, \
           test_labels_id, test_features, test_sentences
