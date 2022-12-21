import src.modelling.NER_module.path as path


def remove_stop_words(sen):
    with open(path.stop_words_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    stop_words = [w.strip() for w in lines]
    splited = sen.split(' ')
    res = ''
    for tok in splited:
        if tok not in stop_words:
            res += tok
            res += ' '
    return res[:-1]
