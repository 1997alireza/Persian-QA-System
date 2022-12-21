def unify_spelling_of_words(sentences, word2idx, idx2word):
    res = []
    for s in sentences:
        splited = s.split(' ')
        new_s = ""
        for tok in splited:
            new_s += (idx2word[word2idx[tok]] + " ")
        res.append(new_s[:-1])
    return res



letters = [('آ', 'ا'), ('ا', 'آ'), ('ی', 'ئ'), ('ئ', 'ی')]
insert_letter = ['ئ']

""" computes edit distance of two words but replacing letter in the list letters
 cost 0.1 instead of 1"""


def edit_distance(str1, str2, tuples):
    # w_ins == w_del == w_sub
    l1 = len(str1)
    l2 = len(str2)
    diff = [[0 for i in range(0, l2 + 1)] for j in range(0, l1 + 1)]
    diff[0][0] = 0
    for i in range(1, l1 + 1):
        diff[i][0] = i
    for i in range(1, l2 + 1):
        diff[0][i] = i
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            # print(str1[i - 1])
            # print(str2[j - 1])
            if str1[i - 1] == str2[j - 1]:
                diff[i][j] = diff[i - 1][j - 1]
            else:
                if (str1[i - 1], str2[j - 1]) in tuples:
                    # print('dfjhjdk')
                    diff[i][j] = min(
                        min(
                            diff[i - 1][j],
                            diff[i][j - 1]
                        ) + 1,
                        diff[i - 1][j - 1]
                        + 0.1
                    )
                else:
                    diff[i][j] = min(
                        diff[i - 1][j],
                        diff[i][j - 1],
                        diff[i - 1][j - 1]

                    ) + 1
                if (str1[i - 1] in insert_letter):
                    # print('fjk')
                    diff[i][j] = min(diff[i][j], diff[i - 1][j] + 0.1)
                if (str2[j - 1] in insert_letter):
                    # print('aa')
                    diff[i][j] = min(diff[i][j], diff[i][j - 1] + 0.1)

    return diff[l1][l2]


"""takes a list of words and returns two dictionaries word2idx and idx2word"""


## TODO: takes long for large lists think of a way to deal with it

def make_dict(words, unify=False, add_unk=False):
    words = list(set(words))
    if (add_unk):
        words = ['unk'] + words
    word2idx = {}
    idx2word = {}
    vocabs = []
    indexes = []
    counter = 1
    for i, w1 in enumerate(words):
        vocabs.append(w1)
        flag = False
        if unify:
            for j, w2 in enumerate(words):
                if i > j and (not flag):
                    # if (len(w1) == len(w2)):
                    # print('dfkjd')
                    sim = edit_distance(w1, w2, letters)
                    # else:
                    #     sim = 5
                    if sim < 1:
                        indexes.append(indexes[j])
                        flag = True
                        break
        if (not flag):
            indexes.append(counter)
            counter += 1
    # print(indexes)
    for i in range(len(vocabs)):
        # print(vocabs[i], indexes[i])
        word2idx[vocabs[i]] = indexes[i]
        if indexes[i] not in idx2word:
            idx2word[indexes[i]] = vocabs[i]
    return word2idx, idx2word
