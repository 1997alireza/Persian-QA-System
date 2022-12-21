import os

root = os.path.dirname(__file__).replace('\\', '/')
if root[-1] != '/':
    root = root + '/'

# directories
src = root + 'src/'
dataset = root + 'dataset/'
saved_models = root + 'saved_models/'

# files
word2vec = dataset + 'simple.wiki.fa.text.vector5'
vocab = saved_models + 'vocab.pkl'
labels = saved_models + 'labels.pkl'
vocab_idf = saved_models + 'vocab_idf_list.pkl'
