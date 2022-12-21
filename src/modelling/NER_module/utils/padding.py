import numpy as np


def pad_and_by_index(sentence_list, tag_list, size, dict_rev, dict_rev2):
    number_of_records = len(sentence_list)
    sentences = np.zeros(shape=(number_of_records, size))
    tags = np.zeros(shape=(number_of_records, size))
    for i in range(number_of_records):
        for j in range(min(size, len(sentence_list[i]))):
            sentences[i][j] = dict_rev[sentence_list[i][j]]
            tags[i][j] = dict_rev2[tag_list[i][j]]
        for j in range(min(size, len(sentence_list[i])), size):
            sentences[i][j] = len(dict_rev) + 1
            tags[i][j] = len(dict_rev2) + 1
    return sentences, tags


def by_index(sentence_list, tag_list, max_size, dict_rev, dict_rev2):
    number_of_records = len(sentence_list)
    sentences = np.zeros(shape=(number_of_records, max_size))
    tags = np.zeros(shape=(number_of_records, max_size))
    for i in range(number_of_records):
        for j in range(min(max_size, len(sentence_list[i]))):
            sentences[i][j] = dict_rev[sentence_list[i][j]]
            tags[i][j] = dict_rev2[tag_list[i][j]]
        for j in range(min(max_size, len(sentence_list[i])), max_size):
            sentences[i][j] = 0
            tags[i][j] = 0
    return sentences, tags


def refine_seq_and_get_lengths(seqs, crop_at=None):
    lens = []
    new_seqs = []
    for seq in seqs:
        if crop_at is not None and len(seq) > crop_at:
            seq = seq[:crop_at]
        i = 1
        while i < len(seq) and seq[-1 * i] == 0:
            i += 1
        for j in range(len(seq)):
            if seq[j] == -1:
                seq[j] = 0
        lens.append(len(seq) - i + 1)

        new_seqs.append(seq)
    return new_seqs, lens
