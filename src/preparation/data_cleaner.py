from src.utils.tools import get_relation_details
from src.utils.graph_extractor import get_top_related_nodes_iri_of_a_relation, get_node_name
import random
import pandas as pd
import paths
import os.path


def final_dataset_generator(rel_question_source=paths.dataset + 'rel_q.csv', append=False):
    """
    to generate the final dataset which is usable for models to train

    @param rel_question_source: a file contains the pairs of relations and questions
    @param append: append new data to the last ones. if it's false, last dataset will be erased!
    """
    full_dataset_address = paths.dataset + 'rel_q_expanded.csv'
    not_exist = not os.path.isfile(full_dataset_address)

    main_file = open(rel_question_source, 'r', encoding='utf8')
    res_file = open(full_dataset_address, 'a' if append else 'w', encoding='utf8')

    if not_exist or not append:
        res_file.write(
            'label(relation),template,template_ner,question,relation iri, start entity iri, end entity iri, answer head')

    last_label = None
    label_iri = ""
    rel_start_ent_key = None
    rel_end_ent_key = None
    top_related_nodes_iri = None

    for row in main_file.readlines()[1:]:
        splitted = row.split(',')
        label = splitted[0]
        main_temp = splitted[1].replace('\n', '')
        if last_label != label and label is not None:
            last_label = label
            try:
                label_iri, rel_start_ent_key, rel_end_ent_key = get_relation_details(label)
            except AssertionError as e:
                last_label = None
                print(e)
                continue
            top_related_nodes_iri = get_top_related_nodes_iri_of_a_relation(label_iri)

        selected_related_nodes_iri = random.sample(top_related_nodes_iri, k=10)  # top 10
        for r_nodes_iri in selected_related_nodes_iri:
            s_ent_name = get_node_name(r_nodes_iri['start'])
            e_ent_name = get_node_name(r_nodes_iri['end'])

            s_ent_word_count = len(s_ent_name.split(' '))
            e_ent_word_count = len(e_ent_name.split(' '))

            question = main_temp.replace(rel_start_ent_key, s_ent_name)
            new_temp = main_temp.replace(rel_start_ent_key, (rel_start_ent_key + ' ') * s_ent_word_count)\
                .replace('  ', ' ')
            question = question.replace(rel_end_ent_key, e_ent_name)
            new_temp = new_temp.replace(rel_end_ent_key, (rel_end_ent_key + ' ') * e_ent_word_count) \
                .replace('  ', ' ')

            ner_tags = get_template_ner(new_temp, [rel_start_ent_key, rel_end_ent_key])


            if rel_start_ent_key in main_temp:
                if rel_end_ent_key in main_temp:
                    answer_head_status = 'unk (both appeared)'  # checked: there isn't any question with this status
                else:
                    answer_head_status = 'end'
            else:
                if rel_end_ent_key in main_temp:
                    answer_head_status = 'start'
                else:
                    answer_head_status = 'unk (none appeared)'  # checked: there isn't any question with this status


            res_file.write('\n' + label + ',' + new_temp + ',' + ner_tags + ',' + question + ',"' +
                           label_iri + '","' + r_nodes_iri['start'] + '","' + r_nodes_iri['end'] + '",' +
                           answer_head_status)

            # print('-------------------------')
            # print('q: ', question)

    main_file.close()
    res_file.close()


def get_template_ner(template, entity_key_list=None):
    # note that it doesn't consider the difference between beginning (`b`) and inner (`i`) tags
    # and generates 'i' for both
    temp_splitted = template.split()
    temp_ner = ''

    if entity_key_list is None:
        is_entity = lambda w: 3 <= len(w) <= 6 and w[0] == w[1] == w[2]
    else:
        is_entity = lambda w: w in entity_key_list

    for t in temp_splitted:
        if is_entity(t):
            temp_ner = temp_ner + 'i'
        else:
            temp_ner = temp_ner + 'o'
    return temp_ner


def split_dataset_randomly(dataset_file_adr=paths.dataset+'rel_q_expanded.csv',
                           splitted_datasets_dir=paths.dataset+'splitted_sets/'):
    dataset = open(dataset_file_adr, 'r', encoding='utf8')
    dataset_lines = dataset.readlines()[1:]
    dataset.close()

    # train, validation, test = 4, 1, 1
    test_and_validation_size = int(len(dataset_lines) * 1 / 6)
    train_size = len(dataset_lines) - 2 * test_and_validation_size

    train_set = open(splitted_datasets_dir + 'train.csv', 'w+', encoding='utf8')
    for i in range(train_size):
        r = int(random.random() * len(dataset_lines))
        train_set.write(dataset_lines.pop(r).replace('\n', '') + '\n')
        # removing and adding '\n' are necessary, are preventing a bug
    train_set.close()

    validation_set = open(splitted_datasets_dir + 'validation.csv', 'w+', encoding='utf8')
    for i in range(test_and_validation_size):
        r = int(random.random() * len(dataset_lines))
        validation_set.write(dataset_lines.pop(r).replace('\n', '') + '\n')
    validation_set.close()

    test_set = open(splitted_datasets_dir + 'test.csv', 'w+', encoding='utf8')
    while len(dataset_lines) > 0:
        r = int(random.random() * len(dataset_lines))
        test_set.write(dataset_lines.pop(r).replace('\n', '') + '\n')
    test_set.close()

    delete_last_empty_line(splitted_datasets_dir + 'train.csv')
    delete_last_empty_line(splitted_datasets_dir + 'validation.csv')
    delete_last_empty_line(splitted_datasets_dir + 'test.csv')


def split_dataset_distinct_randomly(dataset_file_adr=paths.dataset+'rel_q_expanded.csv',
                                    splitted_datasets_dir=paths.dataset+'splitted_distinct_sets/'):
    dataset_csv = pd.read_csv(dataset_file_adr).iloc[:, :].values.tolist()

    # train, validation, test ~= 4, 1, 1
    train_len, validation_len, test_len = 0, 0, 0
    train_temps, valtest_temps = [], []

    train_set = open(splitted_datasets_dir + 'train.csv', 'w+', encoding='utf8')
    validation_set = open(splitted_datasets_dir + 'validation.csv', 'w+', encoding='utf8')
    test_set = open(splitted_datasets_dir + 'test.csv', 'w+', encoding='utf8')

    entities_key = get_entities_key()

    while len(dataset_csv) > 0:
        r = int(random.random() * len(dataset_csv))
        line = dataset_csv.pop(r)
        temp = remove_keys(line[1], entities_key)
        to_write = \
            line[0] + ',' + line[1] + ',' + line[2] + ',' + line[3] + ',"' + line[4] + '","' + line[5] + '","' + \
            line[6] + '",' + line[7] + '\n'

        if train_len / 4 < validation_len or train_len / 4 < test_len:  # try to add to train file
            add_to_train = True
            if temp in valtest_temps:
                add_to_train = False
        else:
            add_to_train = False
            if temp in train_temps:
                add_to_train = True

        if add_to_train:
            train_temps.append(temp)
            train_set.write(to_write)
            train_len += 1

        else:
            valtest_temps.append(temp)
            if validation_len < test_len:
                validation_set.write(to_write)
                validation_len += 1
            else:
                test_set.write(to_write)
                test_len += 1


    train_set.close()
    validation_set.close()
    test_set.close()

    delete_last_empty_line(splitted_datasets_dir + 'train.csv')
    delete_last_empty_line(splitted_datasets_dir + 'validation.csv')
    delete_last_empty_line(splitted_datasets_dir + 'test.csv')


def remove_keys(sentence, entities_key):
    for k in entities_key:
        sentence = sentence.replace(k, '')
        sentence = " ".join(sentence.split())  # remove multiple spaces
    return sentence


def get_entities_key():
    """
    return the set of arguments which are used instead of named entities in question templates.
    each of them usually contains three same characters.
    """
    return set(sum(pd.read_csv(paths.dataset + 'rel_name2uri.csv').iloc[:, 2:4].values.tolist(), []))
    # sum(., []): convert 2d array to 1d


def convert_to_unidirectional_relation(csv_dataset_adr, csv_header=None, new_dataset_address=None):
    """

    :param csv_dataset_adr:
    :param csv_header: can be None (when there is not a row of columns' names) or 'infer'
    :param new_dataset_address:
    :return:
    """
    dataset = pd.read_csv(csv_dataset_adr, header=csv_header)
    adr_split = csv_dataset_adr.split('/')
    if new_dataset_address is None:
        new_dataset_address = '/'.join(adr_split[:-1] + ['unidirectional', adr_split[-1]])
    unidir_dataset = open(new_dataset_address, 'w+', encoding='utf-8')
    for row_id in range(len(dataset)):
        row = dataset.iloc[row_id, :].values.tolist()
        rel = row[0]
        if row[-1] == 'start':
            rel = rel + '0'
        else:
            rel = rel + '1'

        unidir_dataset.write(rel + ',' + row[1] + ',' + row[2] + ',' + row[3] + ',"' + row[4]
                             + '","' + row[5] + '","' + row[6] + '",' + row[7] + '\n')

    unidir_dataset.close()

    delete_last_empty_line(csv_dataset_adr)


def delete_last_empty_line(file_adr):
    file = open(file_adr, 'r', encoding='utf8')
    lines = file.readlines()
    file.close()

    file = open(file_adr, 'w', encoding='utf8')
    for i in range(len(lines)-1):
        file.write(lines[i])
    if lines[-1][-1] == '\n':
        file.write(lines[-1][:-1])
    else:
        file.write(lines[-1])


def test_dataset_generator(test_frac=.1):
    ds = pd.read_csv(paths.dataset + 'rel_q_expanded.csv')

    nums = len(ds)
    test_nums = int(nums * test_frac)

    test_ds = open(paths.dataset + 'end2end_test_dataset.csv', 'w+', encoding='utf8')
    test_ds.write('question,answer entity\n')

    for i in range(test_nums):
        idx = random.randint(0, nums - 1)
        row = ds.iloc[idx, :]
        if row[7] == 'end':
            ans = row[6]
        else:
            ans = row[5]
        test_ds.write(row[3] + ',"' + ans + '"\n')

    test_ds.close()
