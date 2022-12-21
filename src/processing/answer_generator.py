from src.utils.graph_extractor import get_end_entities
from src.processing.classifier_learner import get_trained_rel_model
from src.utils.tools import get_relation_uri, persian_sentence_refinement, get_single_entities_iri
from src.processing.NER_provider import get_tag_extractor


def get_answer_generator(log=False):
    """
    it returns a function (answer generator) which takes a sentence and returns some information.

    """
    tag_extractor = get_tag_extractor()

    label_detector = get_trained_rel_model(model_specification='CNN'  # 'model_specification': 'SVM word2vec'}
                                           )

    def answer_generator(question,
                         multiple_relations=False, check_all_question_entities=False, return_intermediate_info=False,
                         relation_min_prob=.02, relation_prob_difference_threshold=1., relation_count_threshold=3):
        """
        it returns a dictionary containing the detected relation, the posted query and the results of the query.
        if `multiple_relations` is true, instead of just one dictionary, the generator returns an array of dictionaries which are sorted bases on the relations' probabilities.
        if `check_all_question_entities` is true, it checks not only the first qualified question entity, but also all of them.
        if `return_intermediate_info` is true, the result contains intermediate information.
        two threshold and min prob parameters are used to limit the length of the dictionaries array (to return only better results).
        set relation_prob_difference_threshold=1.0 to not consider this threshold

        :return:
            multiple_relations ?
                [{'rel', 'prob', 'result': *result_list}]
                :
                {'rel', 'result': *result_list}

            * result_list = [{'q_word':single_ent_word, 'q_iri': single_ent_iri, 'answers': **answers, 'query': query?)}]
            * if `check_all_question_entities` is False, result_list's length is at most 1
            ** answers' length is at least 1

        """
        if len(question.split()) == 0:
            return None

        question = persian_sentence_refinement(question)
        prediction = label_detector(question, multiple_relations)
        single_entities, detected_entity_words = \
            get_single_entities_iri(question, tag_extractor, return_word_pairs=True, ner_log=log)

        if log:
            print('relation prediction is done: ', prediction)
            print('detected named entities:', single_entities)
            print('detected entity words:', detected_entity_words)


        if multiple_relations:
            sorted_rel_probs = prediction
            sorted_answers = []
            best_prob = sorted_rel_probs[0]['prob']
            for i in range(min(len(sorted_rel_probs), relation_count_threshold)):
                rel, prob = sorted_rel_probs[i]['rel'], sorted_rel_probs[i]['prob']
                if prob < relation_min_prob or best_prob - prob > relation_prob_difference_threshold:
                    break

                result_list = get_answers(rel, single_entities, return_intermediate_info, check_all_question_entities)
                single_answer = {
                    'rel': rel,
                    'prob': prob,
                    'result': result_list,
                }

                sorted_answers.append(single_answer)

            ret_ans = sorted_answers

        else:
            rel = prediction
            result_list = get_answers(rel, single_entities, return_intermediate_info, check_all_question_entities)
            single_answer = {
                'rel': rel,
                'result': result_list
            }

            ret_ans = single_answer

        ret_info = {'ret_ans': ret_ans}
        if return_intermediate_info:
            ret_info['detected_words'] = detected_entity_words
            ret_info['detected_entities'] = single_entities

        return ret_info

    return answer_generator



def get_answers(relation_label, single_entities, return_query=False, check_all_question_entities=False):
    """
    it returns an array of answer entities and the posted queries.
    :return: [{'q_word':single_ent_word, 'q_iri': single_ent_iri, 'answers': answers, 'query': query?)}]
        * answers' length is at least 1
        * if `check_all_question_entities` is False, the array's length is at most 1
    """

    if relation_label[-1] == '0' or relation_label[-1] == '1':
        direction_code = relation_label[-1]
        relation_label = relation_label[:-1]
    else:
        direction_code = None

    relation_uri = get_relation_uri(relation_label)

    checked_answers_set, result_list = [], []  # are used when `check_all_question_entities` is true

    for single_ent in single_entities:
        try:
            answers, query = get_end_entities(single_ent['iri'], relation_uri, direction_code)

            result_item = {'q_word': single_ent['word'], 'q_iri': single_ent['iri'], 'answers': answers}
            if return_query:
                result_item['query'] = query

            if check_all_question_entities:
                answers_set = set(answers)
                if answers_set not in checked_answers_set:
                    checked_answers_set.append(answers_set)
                    result_list.append(result_item)
            else:
                return [result_item]
        except Exception:
            pass

    return result_list
