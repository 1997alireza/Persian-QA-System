from src.modelling.NER_module.NER import recognizer as ner
from src.modelling.NER_module import path as ner_path
from src.modelling.NER_module.models.LSTMCRF import LSTMCRF
from src.modelling.NER_module.models.word2vec import load_word2vec
import re


filter_list = [
    'اسم',
    'کشور',
    'شهر',
    'استان',
    'پایتخت',
    'مکان',
    'بنا',
    'اقیانوس',
    'دریا',
    'ارگان',
    'شرکت',
    'واحد',
    'پول',
    'حکومت',
    'زبان',
    'وزارتخانه',
    'ناشر',
    'کتاب',
    'اثر',
    'فیلم',
    'سریال',
    'تلویزیونی',
    'تیم',
    'ورزشی',
    'ورزشگاه',
    'باشگاه',
    'دوره',
    'زمانی',
    'ملیت',
    'وبسایت',
    'وب‌سایت',
    'تاریخ',
    'کارگردان',
    'نویسنده',
    'بازیگر',
    'آهنگساز',
    'وزیر',
    'ورزشکار',
    'استاندار',
    'سرمربی',
    'شهردار',
    'مدیر',
    'عامل',
    'استاندار',

    'اداره',
    'بیمارستان',
    'تیم',
    'ببرید',
    'کجاست',
    'دوره',
    'کیست',
    'چیست',
    'نشر',
    'کدامین',
    'وسعت',
    'چقدر',
    'شهرداری',
    'سازمان',
    'بانک'
    ]


def standardize_tag_array(tag_array):
    # 'b': beginning, 'i': inside, 'o': outside (not an entity)
    return [tag[0].lower() for tag in tag_array]


def farsbase_tag_extractor(entity):
    tag = entity['nerType']
    if entity['main_class']:
        tag += ' :' + entity['main_class']
    return tag


def get_named_entities(entities, tag_array, consider_inners=False, filter_flag=True):
    """

        :param entities:
        :param tag_array:
        :param filter_flag: should check the filter list ?
        :param consider_inners: should consider start inners as beginners ?
        :return: a list of detected entities (may be they're multi words)
        """
    s_tag_array = standardize_tag_array(tag_array)  # means Small or Standard tag array

    # filtered_tag_array = tag_array.copy()

    named_entities = []

    tag_id = 0
    while tag_id < len(s_tag_array):

        # also check entities that maybe are stating with `i` by the NER module mistake
        if s_tag_array[tag_id] == 'b' or (s_tag_array[tag_id] == 'i' and consider_inners):
            entity = []
            while True:  # a do-while
                if (not filter_flag or entities[tag_id]['word'] not in filter_list):
                    entity.append(entities[tag_id])
                # else:
                #     filtered_tag_array[tag_id] = '-'
                tag_id += 1
                if tag_id >= len(s_tag_array) or s_tag_array[tag_id] != 'i':
                    tag_id -= 1
                    # it's going to summed up by one but the current tag_id shouldn't be ignored (maybe it's 'b')
                    break
            if len(entity) > 0:
                named_entities.append(entity)
        tag_id += 1

    return named_entities


def get_tag_extractor(array_input=False):
    # The NER model cuts the sentence (from the end) if the number of its words is bigger than the model max size.

    word2vec = load_word2vec(ner_path.Word2Vec_file_path)
    parsed_data_adr = ner_path.NER_paresed_datas_path + "/QA_ner_union"
    _, _, dict, dict2, dict_rev, dict_rev2 = ner.load_data(parsed_data_adr)
    dicts = (dict, dict2, dict_rev, dict_rev2)
    model = ner.get_model_object(dicts, ner_path.saved_models + "/all_final", LSTMCRF, word2vec)

    if array_input:
        return lambda sentences: ner.get_tag(sentences, model, dict2, dict_rev)
    else:
        return lambda sentence: ner.get_tag([sentence], model, dict2, dict_rev)[0]


def fix_ner_module_tags(question, entities, ner_module_tags):
    """
    sometimes a space between two words is inferred as a semi-space ('\u200c') when getting entities from FarsBase,
    so the words around that become one word
    """
    words = question.split()
    entities_words = [e['word'] for e in entities]

    if len(ner_module_tags) <= len(entities):
        return ner_module_tags

    new_ner_module_tags = []
    # new_words = []
    while len(entities_words) > 0:
        entity_word = entities_words.pop(0)
        new_word = words.pop(0)
        tag = ner_module_tags.pop(0)
        # while entity_word != new_word:
        # while len(entity_word) > len(new_word):
        while len(re.sub(r"[ \u200c]", "", entity_word)) > len(new_word):
            new_word = new_word + '\u200c' + words.pop(0)
            tag = get_higher_priority_tag(tag, ner_module_tags.pop(0))

        # new_words.append(new_word)
        new_ner_module_tags.append(tag)

    return new_ner_module_tags


def get_higher_priority_tag(t1, t2):
    st1, st2 = standardize_tag_array([t1, t2])
    if st1 == st2:
        return t1
    if st1 == 'b':
        return t1
    if st2 == 'b':
        return t2
    if st1 == 'i':
        return t1
    if st2 == 'i':
        return t2
    return t1
