import pandas as pd
from src.processing.NER_provider import get_tag_extractor
import paths


def evaluation(test_dataset=paths.dataset+'splitted_distinct_sets/test.csv'):
    dataset = pd.read_csv(test_dataset, header=None)

    questions = dataset.iloc[:, 3].values.tolist()
    tag_extractor = get_tag_extractor(array_input=True)
    array_of_tags = tag_extractor(questions)

    wrong_f = open('../../logs/ner_evaluation wrong_augmented_answers.txt', 'w+', encoding='utf8')
    wrong_f.write('<question>, <target tag>, <detected tag>\n')

    correct_phrases, correct_augmented_phrases, TP, FP, FN = 0, 0, 0, 0, 0
    sum_tag_accuracy = 0
    for row_id in range(len(dataset)):
        print('{}/{}'.format(row_id, len(dataset)))
        row = dataset.iloc[row_id, :].values.tolist()
        target_phrase_tag = row[2].replace('b', 'i')
        phrase_tag = array_of_tags[row_id]

        phrase_tag = (''.join([t[0].lower() for t in phrase_tag])).replace('b', 'i')

        # phrase level
        if phrase_tag == target_phrase_tag:
            correct_phrases += 1
            correct_augmented_phrases += 1
        else:
            augmented_phrase_tag = phrase_tag
            diff = max(0, len(target_phrase_tag) - len(phrase_tag))
            augmented_phrase_tag = augmented_phrase_tag + 'o' * diff
            if augmented_phrase_tag == target_phrase_tag:
                correct_augmented_phrases += 1
            else:
                wrong_f.write('{}, {}, {}\n'.format(questions[row_id], target_phrase_tag, phrase_tag))

        correct_tags = 0

        # tag level
        for i in range(len(target_phrase_tag)):
            if i >= len(phrase_tag):
                # not generated tags are considered as `o`
                for j in range(i, len(target_phrase_tag)):
                    if target_phrase_tag[i] == 'o':
                        correct_tags += 1
                    else:
                        FN += 1
                break
            if target_phrase_tag[i] == phrase_tag[i]:
                correct_tags += 1
                if not target_phrase_tag[i] == 'o':
                    TP += 1
            else:
                if target_phrase_tag == 'o':
                    FP += 1
                else:
                    FN += 1

        sum_tag_accuracy += correct_tags / len(target_phrase_tag)

    tag_accuracy = sum_tag_accuracy / len(dataset)
    phrase_accuracy = correct_phrases / len(dataset)
    augmented_phrase_accuracy = correct_augmented_phrases / len(dataset)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)

    print('phrase level: accuracy={}, accuracy(for augmented phrases)={}'
          .format(phrase_accuracy, augmented_phrase_accuracy))
    print('tag level: accuracy={}, precision={}, recall={}, f1_score={}'
          .format(tag_accuracy, precision, recall, f1_score))
