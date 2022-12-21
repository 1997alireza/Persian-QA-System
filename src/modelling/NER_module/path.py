import os

root_directory = os.path.dirname(__file__)
root_directory = root_directory.replace('\\', '/')
saved_models = root_directory + "/saved_models"

bijan_files_dir = os.path.join(root_directory, 'POS/POSData/unannotated')

mohaverekhan_parsed_dir = os.path.join(root_directory, 'models/normalizers/mohaverekhan_parsed')
LOC_entity_path = root_directory + "/Data/NER_lists"
ITRC_NER_rawdata_path = root_directory + "/Data/ITRC_NER_Data"
Word2Vec_file_path = os.path.join(root_directory, 'models/simple_w2v')
Word2Vec_file_path_char = os.path.join(root_directory, 'models/char_w2v')

NER_paresed_datas_path = root_directory + "/NER/parsed_data"
xlsx_NER_data_raw_file = NER_paresed_datas_path + "/AllTagsNERData/NER_ALL_TAGS.xlsx"
NER_data_raw_file = NER_paresed_datas_path + "/NERData/NER.txt"

stop_words_path = root_directory + "/models/normalizers/stopwords.txt"

NER_dataset_path = NER_paresed_datas_path + '/NERData'
all_tags_NER_dataset_path = NER_paresed_datas_path + '/AllTagsNERData'
prefixes = root_directory + '/models/normalizers/prefixes.txt'
suffixes = root_directory + '/models/normalizers/suffixes.txt'
unigram_path = root_directory + '/models/normalizers/wordbreak/unigram.txt'
bigram_path = root_directory + '/models/normalizers/wordbreak/bigram.txt'
added_words_path = root_directory + '/models/normalizers/wordbreak/added_words.txt'

NER_MTL_dataset = root_directory + '/MTL/NER_MTL_Data'
POS_MTL_dataset = root_directory + '/MTL/POS_MTL_data'
POS_raw_files = root_directory + '/POS/POSData/unannotated'



#
# ITRC_dataset_path = NER_paresed_datas_path + "/ITRC_NER_Data"
#
# xlsx_plus_ITRC_path = NER_paresed_datas_path + "/ITRC_plus_ALLTags"
# x1_path = xlsx_plus_ITRC_path + "/x1"
# x2_path = xlsx_plus_ITRC_path + "/x2"
# y1_path = xlsx_plus_ITRC_path + "/y1"
# y2_path = xlsx_plus_ITRC_path + "/y2"
#
# our_data_with_common_tags_path = NER_paresed_datas_path + "/ours_with_common"
# ITRC_data_with_common_tags_path = NER_paresed_datas_path + "/ITRC_with_common"
# ours_and_ITRC_data_with_our_tags_path = NER_paresed_datas_path + "/ITRC_and_ours_with_ours"
# ours_and_ITRC_data_with_common_tags_path = NER_paresed_datas_path + "/ITRC_and_ours_with_common"
