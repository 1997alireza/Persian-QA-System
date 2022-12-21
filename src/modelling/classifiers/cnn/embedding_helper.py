import numpy as np


def read_glove_vectors(glove_vector_path):
    """
    Method to read glove vectors and return an embedding dict.
    returns the embedding matrix and embedding length
    """
    embeddings_index = {}
    with open(glove_vector_path, 'r', encoding='utf8') as f:
        lines = f.read().split('\n')[1:-1]
        for line in lines:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs[:]

    a_word = list(embeddings_index.keys())[0]
    embedding_dim = len(embeddings_index[a_word])
    return embeddings_index, embedding_dim
