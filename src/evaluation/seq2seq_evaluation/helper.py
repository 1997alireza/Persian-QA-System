from src.utils.graph_extractor import generate_query
import pandas as pd


def generate_seq2seq_dataset_from_unidirectional_dataset(csv_dataset_adr, result_directory):
    csv_header = None  # there is not a row of columns' names
    dataset = pd.read_csv(csv_dataset_adr, header=csv_header)
    if result_directory[-1] != '/':
        result_directory = result_directory + '/'
    src = open(result_directory + 'src.txt', 'w+', encoding='utf-8')
    tgt = open(result_directory + 'tgt.txt', 'w+', encoding='utf-8')
    rel = open(result_directory + 'rel.txt', 'w+', encoding='utf-8')
    ans = open(result_directory + 'ans.txt', 'w+', encoding='utf-8')
    que = open(result_directory + 'que.txt', 'w+', encoding='utf-8')

    for row_id in range(len(dataset)):
        row = dataset.iloc[row_id, :].values.tolist()
        question = row[3]
        question_entity_uri = row[5]
        relation_uri = row[4]
        answer_entity_uri = row[6]

        query = generate_query(question_entity_uri, relation_uri, answer_at_end=row[-1] != 'start')
        query = query.replace('\n', '')

        src.write(question + '\n')
        tgt.write(query + '\n')
        rel.write(relation_uri + '\n')
        ans.write(answer_entity_uri + '\n')
        que.write(question_entity_uri + '\n')

    src.close()
    tgt.close()
    rel.close()
    ans.close()
    que.close()


def replace_question_entity(query, entity_iri):
    try:
        first_brace_idx = query.find('{')
        if query[first_brace_idx+1] == '?':  # question entity is at the end
            first_closing_angle_brace = query.find('>')
            o_idx = query.find('<', first_closing_angle_brace + 1)
            c_idx = query.find('>', first_closing_angle_brace + 1)
        else:
            o_idx = query.find('<')
            c_idx = query.find('>')

        query = query[:o_idx+1] + entity_iri + query[c_idx:]

    finally:
        return query


def inverse_query(query):
    first_brace_idx = query.find('{')
    if query[first_brace_idx + 1] == '?':  # question entity is at the end
        first_closing_angle_brace = query.find('>')
        first_opening_angle_brace = query.find('<')
        second_closing_angle_brace = query.find('>', first_closing_angle_brace + 1)

        return query[:first_brace_idx+1] + change_uris(query[first_opening_angle_brace: second_closing_angle_brace+1]) + \
               ' ' + query[first_brace_idx+1: first_opening_angle_brace] +query[second_closing_angle_brace+1:]

    else:
        end_point_idx = query.rfind('.')
        first_closing_angle_brace = query.find('>')
        first_opening_angle_brace = query.find('<')
        second_closing_angle_brace = query.find('>', first_closing_angle_brace + 1)

        return query[:first_brace_idx+1] + query[second_closing_angle_brace+1: end_point_idx] + ' ' + \
               change_uris(query[first_opening_angle_brace: second_closing_angle_brace+1]) + query[end_point_idx:]


def change_uris(sub_query):
    """
    like: `<http://xxx> <http://yyy>` -> `<http://yyy> <http://xxx>`

    """
    first_closing_angle_brace = sub_query.find('>')
    second_opening_angle_brace = sub_query.find('<', first_closing_angle_brace + 1)
    second_closing_angle_brace = sub_query.find('>', first_closing_angle_brace + 1)
    return sub_query[second_opening_angle_brace: second_closing_angle_brace+1] + ' ' + \
           sub_query[0: first_closing_angle_brace+1]

