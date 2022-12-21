import pickle
from os import listdir
from os.path import isfile, join
from xml.dom import minidom



def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def read_bijan_files(dir):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    all_token_tags = dict()
    for file in onlyfiles:
        print(file)
        xmldoc = minidom.parse(dir + '/' + file)
        itemlist = xmldoc.getElementsByTagName('PTB')
        children = itemlist[0].childNodes
        for c in children:
            children2 = c.childNodes
            if len(children2) > 1:
                word = children2[0]
                tag = children2[1]
                word_text = word.childNodes[0].nodeValue
                tag_text = (tag.childNodes[0].nodeValue)[0]

                if word_text in all_token_tags:

                    if tag_text in all_token_tags[word_text]:
                        all_token_tags[word_text][tag_text] += 1
                    else:
                        all_token_tags[word_text][tag_text] = 1
                else:
                    all_token_tags[word_text] = dict()
                    all_token_tags[word_text][tag_text] = 1
    return all_token_tags


def read_NER_lists(dir):
    files = [f for f in listdir(dir)]
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    lists = []
    tags = [f[:3] for f in files]
    for i, file in enumerate(onlyfiles):
        list = []
        with open(dir + "/" + file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            list.append(line.replace("\n", ""))
        lists.append(list)

    return lists, tags
