import numpy as np
import src.modelling.NER_module.path as path
import math

res, dictt, dict_rev, counts, words = None, None, None, None, None
loaded = False
number_of_tokens = 950000
cache = []
misses = []
bigrams = None

""" 
Call this method to break your sentence into vocabularies,
 it also breaks words that are not separated but should be
 """


def breaker(sentence):
    sens = sentence.split(' ')
    newsens = [word_break(s) for s in sens]
    ans = ''
    for i, s in enumerate(newsens):
        ans += s
        if (i != len(newsens) - 1):
            ans += ' '
    return ans


def load_dict(path=path.unigram_path):
    with open(path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    words = []
    counts = []
    for line in lines:
        sp = line.split(' ')
        counts.append(int(sp[1]))
        words.append(sp[0])
    dict = {}
    dict_rev = {}
    for i, word in enumerate(words):
        dict[i] = word
        dict_rev[word] = i
    return dict, dict_rev, counts, words


def load_bigram(path=path.bigram_path):
    global bigrams
    bigrams = []
    with open(path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sp = line.split(' ')

        bigrams.append([sp[0], sp[1], (int(sp[2]) / int(sp[3]))])
    # print('Bigrams Loaded')
    return bigrams


def word_break(sen):
    global res, dictt, dict_rev, counts, words, loaded, cache, misses
    if (not loaded):
        dictt, dict_rev, counts, words = load_dict()
        loaded = True
    res = []
    # t = time.time()
    word_break_worker(sen, len(sen), "")

    # print('time', time.time() - t)
    if len(res) == 0:
        res = [sen]
    # print("sen", sen)
    # print("res", res)
    ans = pick_most_probabile(res)
    cache = []
    misses = []
    return ans


" Generates candidate sentences"


def word_break_worker(sen, n, till_now):
    global res
    for i in range(n):
        prefix = sen[:i + 1]
        if (prefix in words):
            if ((i + 1) == n):
                till_now += prefix
                res.append(till_now)
                return
            word_break_worker(sen[i + 1:], n - (i + 1), till_now + prefix + " ")

def get_prob(w1, w2, bigrams):
    global cache
    global misses

    for bigram in cache:
        if (bigram[0] == w1 and bigram[1] == w2):
            return bigram[2]
    for bigram in misses:
        if (bigram[0] == w1 and bigram[1] == w2):
            return (1 / number_of_tokens)
    for bigram in bigrams:
        if (bigram[0] == w1 and bigram[1] == w2):
            cache.append(bigram)
            return bigram[2]
    misses.append([w1, w2])
    return (1.0 / number_of_tokens)




def compute_sentence_prob(sen):
    global bigrams
    prob = 0
    if (bigrams == None):
        bigrams = load_bigram()
    for i in range(len(sen) - 1):
        p = get_prob(sen[i], sen[i + 1], bigrams)
        prob += math.log(p)
    return prob


def pick_most_probabile(sens):
    probs = [compute_sentence_prob(sen.split(' ')) for sen in sens]
    if len(sens) == 0:
        return ""
    return sens[np.argmax(probs)]
