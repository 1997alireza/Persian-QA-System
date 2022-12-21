"""

To evaluate a system which is similar to the implemented system in this project
and is implemented using a sequence-to-sequence neural network. So we can compare our system with it.
"""

from src.evaluation.seq2seq_evaluation.helper import replace_question_entity, inverse_query
from src.utils.graph_extractor import get_results, BadQueryException
from src.utils.tools import get_single_entities_iri
from src.processing.NER_provider import get_tag_extractor
import paths


def rel_evaluation(prediction_adr, correct_relations_adr):
    with open(prediction_adr, 'r', encoding='utf-8') as f:
        preds = f.readlines()
    with open(correct_relations_adr, 'r', encoding='utf-8') as f:
        rels = [l.replace('\n', '') for l in f.readlines()]

    correct_answers = 0
    for i in range(len(rels)):
        if rels[i] in preds[i]:
            correct_answers += 1

    print('accuracy: {} ({}/{})'.format(correct_answers/len(rels), correct_answers, len(rels)))


def answer_entity_evaluation(prediction_adr, correct_answers_adr, question_adr=None, question_ent_iri_file=None,
                             use_valid_question_entity_iri=False, check_inversed_query=False):
    assert use_valid_question_entity_iri and question_ent_iri_file is not None or \
           not use_valid_question_entity_iri and question_adr is not None, 'required parameters are not provided'


    with open(prediction_adr, 'r', encoding='utf-8') as f:
        preds = f.readlines()
    with open(correct_answers_adr, 'r', encoding='utf-8') as f:
        anss = [l.replace('\n', '') for l in f.readlines()]

    if use_valid_question_entity_iri:
        with open(question_ent_iri_file, 'r', encoding='utf-8') as f:
            question_entities_iri = [l.replace('\n', '') for l in f.readlines()]
    else:
        with open(question_adr, 'r', encoding='utf-8') as f:
            questions = [l.replace('\n', '') for l in f.readlines()]


    ner_tag_extractor = get_tag_extractor()

    correct_answers = 0
    try:
        for i in range(0, len(anss)):
            # if i % 50 == 0:
            print(i, '/', len(anss))
            query = preds[i].replace('\n', '')
            if len(query.replace(' ', '')) == 0:
                continue

            if use_valid_question_entity_iri:
                iri = question_entities_iri[i]
                if 'fkg.iust.ac.ir' not in iri:  # it's not a node in the knowledge base
                    iri = None
            else:
                single_entities_iri, _ = get_single_entities_iri(questions[i], ner_tag_extractor)
                if len(single_entities_iri) == 0:
                    iri = None
                else:
                    iri = single_entities_iri[0]  # use the first detected entity

            if iri is None:
                continue

            iri.replace('\n', '')
            modified_query = replace_question_entity(query, iri)

            ans = []
            try:
                ans = get_results(modified_query)
            except BadQueryException:
                pass

            if check_inversed_query and len(ans) == 0:
                try:
                    ans = get_results(inverse_query(modified_query))
                except BadQueryException:
                    pass

            if anss[i] in ans:
                correct_answers += 1

    finally:
        print('accuracy: {} ({}/{})'.format(correct_answers/len(anss), correct_answers, len(anss)))


if __name__ == '__main__':
    seq2seq_data_dir = paths.src + 'evaluation/seq2seq_evaluation/seq2seq_model/data/'

    answer_entity_evaluation(seq2seq_data_dir + 'pred.txt', seq2seq_data_dir + 'test/ans.txt',
                             question_ent_iri_file=seq2seq_data_dir + 'test/que.txt',
                             use_valid_question_entity_iri=True, check_inversed_query=True)

    # answer_entity_evaluation(seq2seq_data_dir + 'pred.txt', seq2seq_data_dir + 'test/ans.txt',
    #                          question_adr=seq2seq_data_dir + 'test/src.txt', check_inversed_query=False)
    #
    # rel_evaluation(seq2seq_data_dir + 'pred.txt', seq2seq_data_dir + 'test/rel.txt')
