import numpy as np
import src.modelling.NER_module.path as path
from src.modelling.NER_module.utils.IO import read_NER_lists, save_obj, load_obj

# TODO: output variable size, cost function of model

""" above are to data sets that are parsed and saved in given addresses use them 
for "path_of_data" parameter methods of this file but if you want to use your
>>NEW DATA SET<< it must be whether in xlsx format ro txt then you use the "save_and_parse_data" 
method which gives some .pkl files (x, y, dict, dict_rev, dict2, dict_rev2), put them all in one directory and use that as "path_of_data" 
 """
log_path = 'Logs/NER'

common_tags = ['i-loc', 'b-loc', 'b-org', 'i-org', 'o', 'i-per', 'b-per', 'i-dte', 'b-dte']

""" ------------------------------------ USER API PART OF THE CODE ------------------------------------"""


""" 
 This function returns Named Entity Type for sentence_list
 setting word2vec_address to None means pre-trained word2vec was not used to train the model at model_address  
 mode = 1 , only using LSTM
 mode = 2 , only using list
 mode = 3 , using both

set word2vec_address to Word2Vec file used for training 
 """

max_len = 3  # len which we try to use LIST NER


def get_model_object(dicts, model_address, model_class, word2vec_object=None):
    (dict, dict2, dict_rev, dict_rev2) = dicts
    if word2vec_object is None:
        model_object = model_class(dict, dict2, dict_rev, dict_rev2, model_address)
    else:
        model_object = model_class(dict, dict2, dict_rev, dict_rev2, model_address, pretrained_w2v=word2vec_object)
    model_object.load_model()
    return model_object


def get_tag(sentence_list, model_object, dict2, dict_rev, improve_answer=False, mode=1):
    NER_lists = []
    if mode == 2 or mode == 3:
        NER_lists, list_tags = read_NER_lists(path.LOC_entity_path)

    if mode == 1 or mode == 3:

        model = model_object

        sentence_list_split = [sent.split() for sent in sentence_list]
        x = lambda a, sen: a if (a in dict_rev) else 'unk'

        # TODO: an unknow token replaced with zero makes alot of trouble to seqlength in model
        func = lambda t: dict_rev[t.strip()] if t in dict_rev.keys() else -1

        sentence_list_split = [[x(t, sent) for t in sent] for sent in sentence_list_split]
        # sentence_list_rev = [[dict_rev[t.strip()] for t in sent] for sent in sentence_list_split]
        sentence_list_rev = [[func(t) for t in sent] for sent in sentence_list_split]
        init_lens = [len(sent) for sent in sentence_list_split]
        for iterator in range(len(init_lens)):
            for i in range(init_lens[iterator], model.timesteps):
                # NOTICE : padding with zero
                sentence_list_rev[iterator].append(0)

        res = model.get_label(sentence_list_rev)

        answer = [[dict2[t + 1] for t in res[i][:init_lens[i]]] for i in range(len(init_lens))]

        if mode == 3:
            answer = tag_from_lists(sentence_list, max_len, answer)
    elif mode == 2:

        init_lens = [len(sent.split()) for sent in sentence_list]
        answer = [['o' for t in range(init_lens[i])] for i in range(len(init_lens))]
        answer = tag_from_lists(sentence_list, max_len, answer)

    if improve_answer:
        answer = improve_output(answer)
    return answer


"""
tag_from_lists will be called from get_tag
"""


def tag_from_lists(sentence_list, max_len, answer):
    NER_lists, list_tags = read_NER_lists(path.LOC_entity_path)
    sl = [sll.split(' ') for sll in sentence_list]
    # print(reversed(range(1, max_len)))
    for l in reversed(range(1, max_len + 1)):
        for i, ans in enumerate(answer):
            if l <= len(sl[i]):
                some_o = False
                for idx in range(l):
                    if ans[idx] is 'o':
                        some_o = True
                for k in range(len(sl[i]) - l + 1):
                    sen = sl[i][k:k + l]
                    sentence = ""
                    for index in range(len(sen)):
                        sentence += " "
                        sentence += sen[index]
                    sentence = sentence[1:]
                    # print(sentence)

                    for iter in range(len(NER_lists)):
                        if (some_o) and (sentence in NER_lists[iter]):
                            # print('hay')
                            answer[i][k] = ('b-' + list_tags[iter])
                            for j in range(k + 1, k + l):
                                answer[i][j] = ('i-' + list_tags[iter])
                            break
    return answer


"""
used if you are using some new dataset, details explained on top of this file
path

this method creates .pkl files and saves them in "directory" param
"""


def save_and_parse_data(raw_data_path, destination, dictionary_creator_parser, data_creator_parser, base=False,
                        shuffle=False):
    if base:

        dict, dict2, dict_rev, dict_rev2 = dictionary_creator_parser(raw_data_path, base)
        x, y = data_creator_parser(raw_data_path, dict_rev, dict_rev2, base)
    else:
        dict, dict2, dict_rev, dict_rev2 = dictionary_creator_parser(raw_data_path)
        x, y = data_creator_parser(raw_data_path, dict_rev, dict_rev2)
    if shuffle:
        perm = np.random.permutation(len(x))
        x = x[perm]
        y = y[perm]

    save_obj(dict, destination + '/dict')
    save_obj(dict2, destination + '/dict2')
    save_obj(dict_rev, destination + '/dict_rev')
    save_obj(dict_rev2, destination + '/dict_rev2')
    save_obj(x, destination + '/x')
    save_obj(y, destination + '/y')


""" ------------------------------------ END OF USER API ------------------------------------ """

"""
Notice: Input must be all in lowercase
"""


def improve_output(seq):
    improved = [i for i in seq]
    for i, ent in enumerate(seq):
        if i == 0:
            if ent[0] == 'i':
                improved[0] = 'b' + ent[1:]
        else:
            if ent[0] == 'i':
                if not ((seq[i - 1] == 'b' + ent[1:]) or (seq[i - 1] == ent)):
                    improved[i] = 'b' + ent[1:]
    return improved


def load_data(path, x_filename='x', y_filename='y'):
    dict = load_obj(path + '/dict')
    dict2 = load_obj(path + '/dict2')
    dict_rev = load_obj(path + '/dict_rev')
    dict_rev2 = load_obj(path + '/dict_rev2')

    x = load_obj(path + '/' + x_filename)
    y = load_obj(path + '/' + y_filename)

    return x, y, dict, dict2, dict_rev, dict_rev2
