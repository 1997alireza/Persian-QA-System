import pandas as pd
import re
from src.processing.NER_provider import get_named_entities, farsbase_tag_extractor, fix_ner_module_tags
from src.utils.graph_extractor import get_entity_graph_detail
import os
import paths


def get_relation_uri(rel_text, name2uri_file_adr=paths.dataset+'rel_name2uri.csv'):
    uri, _, _ = get_relation_details(rel_text, name2uri_file_adr)
    return uri


def get_relation_details(rel_text, name2uri_file_adr=paths.dataset+'rel_name2uri.csv'):
    dataset = pd.read_csv(name2uri_file_adr)
    labels = dataset.iloc[:, 0].values.tolist()  # label=relation text
    uris = dataset.iloc[:, 1].values.tolist()
    start_entities = dataset.iloc[:, 2].values.tolist()
    end_entities = dataset.iloc[:, 3].values.tolist()
    index = labels.index(rel_text)
    uri = uris[index]
    s_ent = start_entities[index]
    e_ent = end_entities[index]
    assert type(uri) is str, "Could not find the relation's uri. " + '    Relation label: ' + rel_text
    return uri, s_ent, e_ent


def extract_dataset_sentences(question_relation_file_adr):
    dataset = pd.read_csv(question_relation_file_adr)

    labels = dataset.iloc[:, 0].values.tolist()  # label=relation
    templates = dataset.iloc[:, 1].values.tolist()

    if len(dataset.iloc[0]) > 2:
        questions = dataset.iloc[:, 3].values.tolist()
        sentences = templates + questions
        labels *= 2
    else:
        sentences = templates

    # sentences = [persian_sentence_refinement(s) for s in sentences]  # uncomment it and train the models again

    return labels, sentences


def persian_sentence_refinement(sentence):
    sentence = re.sub(r"[,.;:?!،()؟]+", " ", sentence)
    sentence = re.sub(r'[\u200c\s]*\s[\s\u200c]*', " ", sentence)
    sentence = re.sub(r'[\u200c]+', "", sentence)
    sentence = re.sub(r'[\n]+', " ", sentence)
    sentence = re.sub(r'[\t]+', " ", sentence)

    sentence.replace("ي", "ی")
    sentence.replace("ك", "ک")

    sentence = sentence.replace("﷼", "ریال")
    sentence = re.sub(r"ء|ء", "ء", sentence)
    sentence = re.sub(r"ﺇ|ﺎ|ا|ا", "ا", sentence)
    sentence = re.sub(r"ب|ﺑ", "ب", sentence)
    sentence = re.sub(r"ة|ۀ|ﮤ", "ه", sentence)
    sentence = re.sub(r"ت|ﺘ", "ت", sentence)
    sentence = re.sub(r" ج|ﺟ", "ج", sentence)
    sentence = re.sub(r"ج|ﺟ", "ج", sentence)
    sentence = re.sub(r"ح", "ح", sentence)
    sentence = re.sub(r"خ|ﺧ", "خ", sentence)
    sentence = re.sub(r"د|ﺪ", "د", sentence)
    sentence = re.sub(r"ذ", "ذ", sentence)
    sentence = re.sub(r"ر|ﺮ", "ر", sentence)
    sentence = re.sub(r"ز|ﺰ", "ز", sentence)
    sentence = re.sub(r"س|ﺳ|ﺴ", "س", sentence)
    sentence = re.sub(r" ش|ﺶ", "ش", sentence)
    sentence = re.sub(r"ص|ﺻ", "ص", sentence)
    sentence = re.sub(r"ض", "ض", sentence)
    sentence = re.sub(r"ط|ﻃ", "ط", sentence)
    sentence = re.sub(r"ظ", "ظ", sentence)
    sentence = re.sub(r"ع", "ع", sentence)
    sentence = re.sub(r"غ|ﻐ", "غ", sentence)
    sentence = re.sub(r"ػ", "ک", sentence)
    sentence = re.sub(r"ؼ", "ک", sentence)
    sentence = re.sub(r"ف|ﻔ", "ف", sentence)
    sentence = re.sub(r"ق|ﻗ", "ق", sentence)
    sentence = re.sub(r"ل", "ل", sentence)
    sentence = re.sub(r"م", "م", sentence)
    sentence = re.sub(r"ه", "ه", sentence)
    sentence = re.sub(r"و", "و", sentence)
    sentence = re.sub(r"ٯ", "و", sentence)
    sentence = re.sub(r"ٱ", "ا", sentence)
    sentence = re.sub(r"ٲ", "ا", sentence)
    sentence = re.sub(r"ٳ", "ا", sentence)
    sentence = re.sub(r"ٴ ", "", sentence)
    sentence = re.sub(r"ٶ", "ؤ", sentence)
    sentence = re.sub(r"ٸ", "ئ", sentence)
    sentence = re.sub(r"ٹ", "ت", sentence)
    sentence = re.sub(r"ټ", "ت", sentence)
    sentence = re.sub(r"ٽ", "ث", sentence)
    sentence = re.sub(r"پ", "پ", sentence)
    sentence = re.sub(r"ٿ", "ت", sentence)
    sentence = re.sub(r"ځ", "ح", sentence)
    sentence = re.sub(r"ڃ", "ج", sentence)
    sentence = re.sub(r"ڄ", "ج", sentence)
    sentence = re.sub(r"څ", "خ", sentence)
    sentence = re.sub(r"چ", "چ", sentence)
    sentence = re.sub(r"ڇ", "چ", sentence)
    sentence = re.sub(r"ڈ", "د", sentence)
    sentence = re.sub(r"ډ", "د", sentence)
    sentence = re.sub(r"ڌ", "ذ", sentence)
    sentence = re.sub(r"ڍ", "د", sentence)
    sentence = re.sub(r"ڑ", "ر", sentence)
    sentence = re.sub(r"ڒ", "ر", sentence)
    sentence = re.sub(r"ړ", "ر", sentence)
    sentence = re.sub(r"ڕ", "ر", sentence)
    sentence = re.sub(r"ږ", "ز", sentence)
    sentence = re.sub(r"ڗ", "ز", sentence)
    sentence = re.sub(r"ژ", "ژ", sentence)
    sentence = re.sub(r"ښ", "س", sentence)
    sentence = re.sub(r"ڞ", "ض", sentence)
    sentence = re.sub(r"ڟ", "ظ", sentence)
    sentence = re.sub(r"ڡ", "ف", sentence)
    sentence = re.sub(r"ڤ", "ف", sentence)
    sentence = re.sub(r"ک", "ک", sentence)
    sentence = re.sub(r"ڪ", "ک", sentence)
    sentence = re.sub(r"ګ", "ک", sentence)
    sentence = re.sub(r"ڭ", "ک", sentence)
    sentence = re.sub(r"گ", "گ", sentence)
    sentence = re.sub(r"ڰ", "گ", sentence)
    sentence = re.sub(r"ڱ", "گ", sentence)
    sentence = re.sub(r"ڵ", "ل", sentence)
    sentence = re.sub(r"ں", "ن", sentence)
    sentence = re.sub(r"ڼ", "ن", sentence)
    sentence = re.sub(r"ھ", "ه", sentence)
    sentence = re.sub(r"ۀ", "ه", sentence)
    sentence = re.sub(r"ہ", "ه", sentence)
    sentence = re.sub(r"ۂ", "ه", sentence)
    sentence = re.sub(r"ۃ", "ه", sentence)
    sentence = re.sub(r"ۅ", "و", sentence)
    sentence = re.sub(r"ۆ", "و", sentence)
    sentence = re.sub(r"ۇ", "و", sentence)

    sentence = re.sub(r"ۈ", "و", sentence)
    sentence = re.sub(r"ۉ", "و", sentence)
    sentence = re.sub(r"ۊ", "و", sentence)
    sentence = re.sub(r"ی", "ی", sentence)
    sentence = re.sub(r"ۍ", "ی", sentence)
    sentence = re.sub(r"ێ", "ی", sentence)
    sentence = re.sub(r"ۏ", "و", sentence)
    sentence = re.sub(r"ې|ﯼ|ﯽ|ﯾ", "ی", sentence)
    sentence = re.sub(r"ے", "ی", sentence)
    sentence = re.sub(r"ۓ", "ی", sentence)
    sentence = re.sub(r"ە", "ه", sentence)
    sentence = re.sub(r"ۥ", "و", sentence)
    sentence = re.sub(r"۽", "", sentence)
    sentence = re.sub(r"ݘ", "چ", sentence)
    sentence = re.sub(r"ݜ", "ش", sentence)
    sentence = re.sub(r"ݡ", "ف", sentence)
    sentence = re.sub(r"ݻ", "ی", sentence)
    sentence = re.sub(r"ﭖ", "پ", sentence)
    sentence = re.sub(r"ﭗ", "پ", sentence)
    sentence = re.sub(r"ﭘ | ﭘ", "پ", sentence)
    sentence = re.sub(r"ﭙ", "پ", sentence)
    sentence = re.sub(r"ﭞ", "ت", sentence)
    sentence = re.sub(r"ﭵ", "ج", sentence)
    sentence = re.sub(r"ﭺ", "چ", sentence)
    sentence = re.sub(r"ﭻ", "چ", sentence)
    sentence = re.sub(r"ﭼ|ﭼ", "چ", sentence)
    sentence = re.sub(r"ﭽ", "چ", sentence)
    sentence = re.sub(r"ﮊ", "ژ", sentence)
    sentence = re.sub(r"ﮋ", "ژ", sentence)
    sentence = re.sub(r"ﮎ|ﮐ|ﮑ|ﻛ", "ک", sentence)
    sentence = re.sub(r"ﮏ", "ک", sentence)
    sentence = re.sub(r"ﮐ", "ک", sentence)
    sentence = re.sub(r"ﮑ", "ک", sentence)
    sentence = re.sub(r"ﻙ|ﻚ", "ک", sentence)
    sentence = re.sub(r"ﮒ", "گ", sentence)
    sentence = re.sub(r"ﮓ|ﮓ", "گ", sentence)
    sentence = re.sub(r"ﮔ", "گ", sentence)
    sentence = re.sub(r"ﮕ|ﮕ", "گ", sentence)
    sentence = re.sub(r"ﮚ", "گ", sentence)
    sentence = re.sub(r"ﮤ|ۀ", "ه", sentence)
    sentence = re.sub(r"ﮥ", "ه", sentence)
    sentence = re.sub(r"ﮧ", "ه", sentence)
    sentence = re.sub(r"ﮩ", "ه", sentence)
    sentence = re.sub(r"ﮫ", "ه", sentence)
    sentence = re.sub(r"ﮬ", "ه", sentence)
    sentence = re.sub(r"ﮭ", "ه", sentence)
    sentence = re.sub(r"ﮮ", "ی", sentence)
    sentence = re.sub(r"ﮯ", "ی", sentence)
    sentence = re.sub(r"ﯚ", "و", sentence)
    sentence = re.sub(r"ﯼ", "ی", sentence)
    sentence = re.sub(r"ﯽ", "ی", sentence)
    sentence = re.sub(r"ﯾ", "ی", sentence)
    sentence = re.sub(r"ﯿ", "ی", sentence)
    sentence = re.sub(r"ﳊ", "لح", sentence)
    sentence = re.sub(r"ﷲ", "الله", sentence)
    sentence = re.sub(r"ﺀ", "ء", sentence)
    sentence = re.sub(r"ﺁ", "آ", sentence)
    sentence = re.sub(r"ﺂ", "آ", sentence)
    sentence = re.sub(r"ﺃ", "أ", sentence)
    sentence = re.sub(r"ﺄ", "أ", sentence)
    sentence = re.sub(r"ﺅ", "ؤ", sentence)
    sentence = re.sub(r"ﺆ", "ؤ", sentence)
    sentence = re.sub(r"ﺈ", "ا", sentence)
    sentence = re.sub(r"ﺉ", "ئ", sentence)
    sentence = re.sub(r"ﺊ", "ئ", sentence)
    sentence = re.sub(r"ﺋ", "ئ", sentence)
    sentence = re.sub(r"ﺌ", "ئ", sentence)
    sentence = re.sub(r"ﺍ", "ا", sentence)
    sentence = re.sub(r"ﺎ", "ا", sentence)
    sentence = re.sub(r"ﺏ", "ب", sentence)
    sentence = re.sub(r"ﺐ", "ب", sentence)
    sentence = re.sub(r"ﺑ", "ب", sentence)
    sentence = re.sub(r"ﺒ", "ب", sentence)
    sentence = re.sub(r"ﺓ", "ه", sentence)
    sentence = re.sub(r"ﺔ", "ه", sentence)
    sentence = re.sub(r"ﺕ", "ت", sentence)
    sentence = re.sub(r"ﺖ", "ت", sentence)
    sentence = re.sub(r"ﺗ", "ت", sentence)
    sentence = re.sub(r"ﺘ", "ت", sentence)
    sentence = re.sub(r"ﺙ", "ث", sentence)
    sentence = re.sub(r"ﺚ", "ث", sentence)
    sentence = re.sub(r"ﺛ", "ث", sentence)
    sentence = re.sub(r"ﺜ", "ث", sentence)
    sentence = re.sub(r"ﺝ", "ج", sentence)
    sentence = re.sub(r"ﺞ", "ج", sentence)
    sentence = re.sub(r"ﺟ", "ج", sentence)
    sentence = re.sub(r"ﺠ", "ج", sentence)
    sentence = re.sub(r"ﺡ", "ح", sentence)
    sentence = re.sub(r"ﺢ", "ح", sentence)
    sentence = re.sub(r"ﺣ", "ح", sentence)
    sentence = re.sub(r"ﺤ", "ح", sentence)
    sentence = re.sub(r"ﺥ", "خ", sentence)
    sentence = re.sub(r"ﺦ", "خ", sentence)
    sentence = re.sub(r"ﺧ", "خ", sentence)
    sentence = re.sub(r"ﺨ", "خ", sentence)
    sentence = re.sub(r"ﺩ", "د", sentence)
    sentence = re.sub(r"ﺪ", "د", sentence)
    sentence = re.sub(r"ﺫ", "ذ", sentence)
    sentence = re.sub(r"ﺬ", "ذ", sentence)
    sentence = re.sub(r"ﺭ", "ر", sentence)
    sentence = re.sub(r"ﺮ", "ر", sentence)
    sentence = re.sub(r"ﺯ", "ز", sentence)
    sentence = re.sub(r"ﺰ", "ز", sentence)
    sentence = re.sub(r"ﺱ", "س", sentence)
    sentence = re.sub(r"ﺲ", "س", sentence)
    sentence = re.sub(r"ﺳ", "س", sentence)
    sentence = re.sub(r"ﺴ", "س", sentence)
    sentence = re.sub(r"ﺵ", "ش", sentence)
    sentence = re.sub(r"ﺶ", "ش", sentence)
    sentence = re.sub(r"ﺷ", "ش", sentence)
    sentence = re.sub(r"ﺸ", "ش", sentence)
    sentence = re.sub(r"ﺹ", "ص", sentence)
    sentence = re.sub(r"ﺺ", "ص", sentence)
    sentence = re.sub(r"ﺻ", "ص", sentence)
    sentence = re.sub(r"ﺼ", "ص", sentence)
    sentence = re.sub(r"ﺽ", "ض", sentence)
    sentence = re.sub(r"ﺾ", "ض", sentence)
    sentence = re.sub(r"ﺿ", "ض", sentence)
    sentence = re.sub(r"ﻀ", "ض", sentence)
    sentence = re.sub(r"ﻁ", "ط", sentence)
    sentence = re.sub(r"ﻂ", "ط", sentence)
    sentence = re.sub(r"ﻃ", "ط", sentence)
    sentence = re.sub(r"ﻄ", "ط", sentence)
    sentence = re.sub(r"ﻅ", "ظ", sentence)
    sentence = re.sub(r"ﻆ", "ظ", sentence)
    sentence = re.sub(r"ﻇ", "ظ", sentence)
    sentence = re.sub(r"ﻈ", "ظ", sentence)
    sentence = re.sub(r"ﻉ", "ع", sentence)
    sentence = re.sub(r"ﻊ", "ع", sentence)
    sentence = re.sub(r"ﻋ", "ع", sentence)
    sentence = re.sub(r"ﻌ", "ع", sentence)
    sentence = re.sub(r"ﻍ", "غ", sentence)
    sentence = re.sub(r"ﻎ", "غ", sentence)
    sentence = re.sub(r"ﻏ", "غ", sentence)
    sentence = re.sub(r"ﻐ", "غ", sentence)
    sentence = re.sub(r"ﻑ", "ف", sentence)
    sentence = re.sub(r"ﻒ", "ف", sentence)
    sentence = re.sub(r"ﻓ", "ف", sentence)
    sentence = re.sub(r"ﻔ", "ف", sentence)
    sentence = re.sub(r"ﻕ", "ق", sentence)
    sentence = re.sub(r"ﻖ", "ق", sentence)
    sentence = re.sub(r"ﻗ", "ق", sentence)
    sentence = re.sub(r"ﻘ", "ق", sentence)
    sentence = re.sub(r"ﻙ", "ک", sentence)
    sentence = re.sub(r"ﻚ", "ک", sentence)
    sentence = re.sub(r"ﻛ", "ک", sentence)
    sentence = re.sub(r"ﻜ", "ک", sentence)
    sentence = re.sub(r"ﻝ", "ل", sentence)
    sentence = re.sub(r"ﻞ", "ل", sentence)
    sentence = re.sub(r"ﻟ", " ل", sentence)
    sentence = re.sub(r" ﻠ|ﻟ|ﻠ", " ل", sentence)
    sentence = re.sub(r"ﻡ", "م", sentence)

    sentence = re.sub(r"ﻡ", "م", sentence)
    sentence = re.sub(r"ﻢ", "م", sentence)
    sentence = re.sub(r"ﻣ", "م", sentence)
    sentence = re.sub(r"ﻤ|ﻣ|ﻤ", "م", sentence)
    sentence = re.sub(r"ﻥ|ﻦ|ﻧ|ﻨ", "ن", sentence)
    sentence = re.sub(r"ﻦ", "ن", sentence)
    sentence = re.sub(r"ﻧ", "ن", sentence)
    sentence = re.sub(r"ﻨ", "ن", sentence)
    sentence = re.sub(r"ﻩ", "ه", sentence)
    sentence = re.sub(r"ﻪ|ﻪ", "ه", sentence)
    sentence = re.sub(r"ﻫ", "ه", sentence)
    sentence = re.sub(r"ﻬ", "ه", sentence)
    sentence = re.sub(r"ﻭ", "و", sentence)
    sentence = re.sub(r"ﻮ|ﻮ", "و", sentence)
    sentence = re.sub(r"ﻰ", "ی", sentence)
    sentence = re.sub(r"ﻱ", "ی", sentence)
    sentence = re.sub(r"ﻲ", "ی", sentence)
    sentence = re.sub(r"ﻳ", "ی", sentence)
    sentence = re.sub(r"ﻴ|ﻴ", "ی", sentence)
    sentence = re.sub(r"ﻵ", "لا", sentence)
    sentence = re.sub(r"ﻷ", "لا", sentence)
    sentence = re.sub(r"ﻸ", "لا", sentence)
    sentence = re.sub(r"ﻻ", "لا", sentence)
    sentence = re.sub(r"ﻼ", "لا", sentence)
    sentence = re.sub(r"ﻶ", "لا", sentence)
    sentence = re.sub(r"ﻹ", "لا", sentence)

    sentence = re.sub(r"０|۰|٠", "0", sentence)
    sentence = re.sub(r"１|۱|١", "1", sentence)
    sentence = re.sub(r"２|۲|٢", "2", sentence)
    sentence = re.sub(r"３|۳|٣", "3", sentence)
    sentence = re.sub(r"４|۴|٤", "4", sentence)
    sentence = re.sub(r"５|۵|٥", "5", sentence)
    sentence = re.sub(r"６|۶|٦", "6", sentence)
    sentence = re.sub(r"７|۷|٧", "7", sentence)
    sentence = re.sub(r"８|۸|٨", "8", sentence)
    sentence = re.sub(r"９|۹|٩", "9", sentence)

    return sentence


def get_single_entities_iri(question, ner_tag_extractor=None, return_word_pairs=False, ner_log=False):
    entities, _ = get_entity_graph_detail(question)

    farsbase_tags = [farsbase_tag_extractor(ent) for ent in entities]

    ner_procedure_list = [[farsbase_tags, False, "KB tags"]]

    if ner_tag_extractor is not None:
        ner_module_tags = ner_tag_extractor(question)
        try:
            ner_module_tags = fix_ner_module_tags(question, entities, ner_module_tags)
            ner_procedure_list = [
                                     [ner_module_tags, False, "NER module tags"],
                                     [ner_module_tags, True, "NER module tags, considering inners"]
                                 ] + ner_procedure_list
        except Exception as e:
            if ner_log:
                print('Problem in `fix_ner_module_tags`: ', e)


    single_entities, single_entities_iri = [], []
    detected_entity_words = set()  # whether with iri or not

    for arguments in ner_procedure_list:
        procedure_name = arguments.pop()
        try:
            named_entities = get_named_entities(entities, *arguments)
            for entity in named_entities:
                for single_ent in entity:
                    detected_entity_words.add(single_ent['word'])
                    if single_ent['iri'] is not None and single_ent['iri'] is not '' \
                            and single_ent['iri'] not in single_entities_iri:
                        single_entities_iri.append(single_ent['iri'])
                        single_entities.append({'word': single_ent['word'], 'iri': single_ent['iri']})

        except Exception as e:
            if ner_log:
                print('Problem in `get_named_entities` on {}: '.format(procedure_name), e)

    if return_word_pairs:
        return single_entities, list(detected_entity_words)
    else:
        return single_entities_iri, list(detected_entity_words)


def get_file_name(path):
    file_base_name = os.path.basename(path)
    return os.path.splitext(file_base_name)[0]
