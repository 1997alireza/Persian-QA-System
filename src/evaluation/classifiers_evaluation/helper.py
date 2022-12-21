from random import random
from src.modelling.classifiers.svm import SVM_Model
from src.modelling.classifiers.cnn.cnn import CNN_Model
import paths


def k_fold_test(vocab, labels_set, features, labels_id, model_class=SVM_Model, k=2):
    print('{}-fold test started'.format(k))
    size_label_id = -1
    for li in labels_id:
        size_label_id = max(size_label_id, li)
    size_label_id += 1

    l = len(features)
    l_k = int(l / k)
    k_indexes = []
    pre_indexes = []
    for i in range(l):
        pre_indexes.append(i)
    for i in range(k):
        index_i = []
        for j in range(l_k):
            index_i.append(pre_indexes.pop(int(random() * len(pre_indexes))))
        k_indexes.append(index_i)

    for i in range(l_k * k, l):
        k_indexes[int(random() * len(k_indexes))].append(pre_indexes.pop())

    sum_correct_percent = 0
    sum_precision = 0
    sum_recall = 0
    for i in range(k):  # test on part i
        print('     --test on part {}'.format(i))
        i_features = []
        i_labels_id = []
        test_i_features = []
        test_i_labels_id = []
        corrects = 0
        incorrects = 0

        for j in range(k):
            if i is not j:
                for x in k_indexes[j]:
                    i_features.append(features[x])
                    i_labels_id.append(labels_id[x])

        for x in k_indexes[i]:
            test_i_features.append(features[x])
            test_i_labels_id.append(labels_id[x])

        model = model_class()
        if model_class == CNN_Model:
            model.initialize(vocab, len(labels_set))

        print("     part's train started")
        model.train(i_features, i_labels_id)

        yUh = [0] * size_label_id  # intersect
        y = [0] * size_label_id
        h = [0] * size_label_id

        predicted_label_ids = model.predict(test_i_features)
        for m in range(len(test_i_features)):
            if m % 100 == 0:
                print('          part {}: test {}/{}'.format(i, m, len(test_i_features)))
            # predicted_label_id = model.predict_one(test_i_features[m])
            predicted_label_id = predicted_label_ids[m]
            real_label_id = test_i_labels_id[m]
            if predicted_label_id < size_label_id:
                h[predicted_label_id] += 1
            y[real_label_id] += 1
            if predicted_label_id == real_label_id:
                corrects += 1
                yUh[predicted_label_id] += 1
            else:
                incorrects += 1

        sum_precision_part = 0
        sum_recall_part = 0
        for li in range(size_label_id):
            precision_label_i = 0
            recall_label_i = 0
            if h[li] > 0:
                precision_label_i = yUh[li] / h[li]
            if y[li] > 0:
                recall_label_i = yUh[li] / y[li]

            sum_precision_part += precision_label_i
            sum_recall_part += recall_label_i

        sum_precision += sum_precision_part / size_label_id
        sum_recall += sum_recall_part / size_label_id

        correct_percent = corrects / (corrects + incorrects)
        sum_correct_percent += correct_percent


    correctness_ratio = sum_correct_percent / k
    precision = sum_precision / k
    recall = sum_recall / k
    f1_score = 2 * precision * recall / (precision + recall)
    return correctness_ratio, precision, recall, f1_score


def test_model_on_sets(
        vocab, labels_set, train_features, train_labels_id, test_features, test_labels_id, test_sentences,
        model_class=SVM_Model):

    model = model_class()
    if model_class == CNN_Model:
        model.initialize(vocab, len(labels_set))
    model.train(train_features, train_labels_id)
    print('model is trained')

    size_label_id = -1
    for li in train_labels_id + test_labels_id:
        size_label_id = max(size_label_id, li)
    size_label_id += 1

    yUh = [0] * size_label_id  # intersect
    y = [0] * size_label_id
    h = [0] * size_label_id
    corrects, incorrects = 0, 0

    log_file = open(paths.root + 'logs/' + model_class.__name__ + ' new.txt', 'w', encoding='utf8')
    log_file.write('sentence, real_label, predicted_label\n')

    predicted_label_ids = model.predict(test_features)

    for m in range(len(test_features)):
        # predicted_label_id = model.predict_one(test_features[m])
        predicted_label_id = predicted_label_ids[m]
        real_label_id = test_labels_id[m]

        if predicted_label_id < size_label_id:
            h[predicted_label_id] += 1
        y[real_label_id] += 1
        if predicted_label_id == real_label_id:
            corrects += 1
            yUh[predicted_label_id] += 1
        else:
            incorrects += 1
            log_file.write('{}, {}, {}\n'.format(test_sentences[m], labels_set[real_label_id], labels_set[predicted_label_id]))

        if m % 100 == 0:
            print('test {} / {}'.format(m, len(test_features)))

    sum_precision = 0
    sum_recall = 0
    for li in range(size_label_id):
        precision_label_i = 0
        recall_label_i = 0
        if h[li] > 0:
            precision_label_i = yUh[li] / h[li]
        if y[li] > 0:
            recall_label_i = yUh[li] / y[li]

        sum_precision += precision_label_i
        sum_recall += recall_label_i

    log_file.close()

    precision = sum_precision / size_label_id
    recall = sum_recall / size_label_id
    correctness_ratio = corrects / (corrects + incorrects)
    f1_score = 2 * precision * recall / (precision + recall)

    return correctness_ratio, precision, recall, f1_score
