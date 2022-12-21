import numpy as np
import time

# simple_dir = 'models/simple_w2v'
simple_dir_org_file = 'Data/twitter_wikipedia_hamshahri_irblog/simple/twitt_wiki_ham_blog.fa.text.100.vec'
char_dir_org_file = 'Data/twitter_wikipedia_hamshahri_irblog/char/twitt_wiki_ham_blog.fa.char.vector100'

from src.modelling.NER_module.utils.IO import save_obj, load_obj

"""parsing word2vec file each time takes time so this function parses word2vec file
    and saves the parsed object into a pkl file and each time pkl file is loaded and
    no parsing is needed so it is much quicker"""


def parse_and_save_word2vec(dir,des_file_name):
    t = time.time()
    with open(dir, encoding='utf-8', mode='r') as f:
        lines = f.readlines()
    line0 = lines[0].split(' ')
    word_num = int(line0[0])
    embedding_size = int(line0[1])
    vocab = {}
    for i in range(1, word_num):
        word = lines[i].split(' ')[0]
        vec = lines[i].split(' ')[1:]
        # print(word)
        # print(vec)
        vec.remove('\n')
        # vec = [float(v) for v in vec]
        vocab[word] = np.array(vec, dtype="float32")
    # print('word2vec parsed in', time.time() - t, 's')
    save_obj(vocab, des_file_name)
    return vocab


def load_word2vec(dir):
    t = time.time()
    vocab = load_obj(dir)
    # print('word2vec loaded in', time.time() - t, 's')
    return vocab
