"""
working with FarsBase knowledge-base
"""

import requests
import json
import xml.etree.ElementTree as ET


class NonResultQueryException(Exception):
    pass


class BadQueryException(Exception):
    pass


def get_all_ambiguities(text):
    url = "http://farsbase.net:8099/proxy/extractor/rest/v1/extractor/extract.json"

    querystring = {"text": text}

    headers = {
        'Cache-Control': "no-cache"
    }

    return requests.request("GET", url, headers=headers, params=querystring, timeout=5).text


def get_unambiguous(text):
    url = "http://farsbase.net:8099/proxy/raw/rest/v1/raw/FKGfy"

    payload = '{"text": "' + text + '"}'
    headers = {
        'Content-Type': "application/json",
        'Authorization': "Basic ZHJfbW9tdGF6aTpAOGQyI0xxSA==",
        'Cache-Control': "no-cache"
    }

    return requests.request("POST", url, data=payload.encode('utf-8'), headers=headers, timeout=5).text


def get_entity_graph_detail(text):
    text = text.replace('.', ' ')  # the FarsBase api read the sentence until a dot character
    resp_json = get_unambiguous(text)
    resp_list = json.loads(resp_json)[0]
    entities = []
    for en in resp_list:
        res = en['resource']
        ent = {'word': en['word'], 'iri': None, 'main_class': None, 'nerType': en['iobType']}
        if res:
            ent['iri'] = res['iri']
            ent['main_class'] = res['mainClass']
        entities.append(ent)

    return entities, resp_json


def post_query(query):
    """

    @raise requests.exceptions.ConnectionError, BadQueryException
    """
    url = "http://farsbase.net:8890/sparql/"

    payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"query\"\r\n\r\n" \
              + query + "\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"
    headers = {
        'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        'Cache-Control': "no-cache"
    }

    resp_xml = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers, timeout=5).text
    # may raise requests.exceptions.ConnectionError exception

    try:
        root = ET.fromstring(resp_xml)
        root_results = root[1]
        return root_results
    except Exception as e:
        raise BadQueryException(e)


def get_results(query):
    """

    @raise requests.exceptions.ConnectionError, BadQueryException
    """
    # TODO: need to know exactly whether FarsBase is returned the answer of the query or other information
    root_results = post_query(query)

    results = []
    for r in root_results:
        try:
            element = r[0][0]
            text = element.text
            # tag = element.tag
            # brace_pos = tag.rfind('}')
            # return tag[brace_pos+1:] == 'uri'
            # -> not ok, maybe it's a literal and it's the correct answer
        except:
            continue

        if text.lower() != 'yes':
            if 'fkg.iust.ac.ir' in text:
                try:
                    pos = text.rfind('/')
                    last_part = text[pos+1:]
                    if last_part[:9] == 'relation_':
                        text = text[:pos]
                except Exception:
                    pass

            results.append(text)

    return results


def get_end_entities(question_entity_uri, relation_uri, direction_code=None):
    answer_at_end = True
    if direction_code == '0':
        answer_at_end = False

    query = generate_query(question_entity_uri, relation_uri, answer_at_end)

    try:
        answers = get_results(query)
        if len(answers) != 0:
            return answers, query
    except Exception:  # (IndexError, KeyError):
        pass

    # the answer entity actually is on the other side of the relation
    alternative_query = generate_query(question_entity_uri, relation_uri, not answer_at_end)

    try:
        answers = get_results(alternative_query)
        if len(answers) != 0:
            return answers, alternative_query
    except Exception:
        pass

    raise NonResultQueryException(
        'Could not find any end entity\n    Requested queries: \n' + query + '\n' + alternative_query)


def generate_query(question_entity_uri, relation_uri, answer_at_end):
    # query = 'select distinct(?s) ?o where {\n' \
    #         '?s ' + '<' + relation_uri + '> ?o.\n' \
    #         '?s ' + start_entity + '.\n}'

    if answer_at_end:  # answer entity is at end, direction_code == '1'
        return 'select distinct(?o) where {\n'\
               '<' + question_entity_uri + '>' + ' <' + relation_uri + '> ?o.\n' \
               '}'

    else:  # answer entity is at start, direction_code == '0'
        return 'select distinct(?o) where {\n'\
               '?o <' + relation_uri + '>' + ' <' + question_entity_uri + '>.\n'\
               '}'


def get_top_entities_of_type(entities_type_uri='http://fkg.iust.ac.ir/ontology/City', limit=10):
    query = 'select  ?s where\n' \
            '{ ?a ?b ?s .\n' \
            '?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#instanceOf> <' + entities_type_uri + '>.\n' \
            '}\n' \
            'group by ?s\n' \
            'order by desc(count(*) as ?cnt)\n' \
            'limit ' + str(limit)

    root_results = post_query(query)
    tops = []
    for res in root_results:
        tops.append(res[0][0].text)
    return tops


def get_top_related_nodes_iri_of_a_relation(relation_uri='http://fkg.iust.ac.ir/ontology/capital', limit=50):
    query = 'select ?start ?end where\n' \
            '{\n' \
            '?start <' + relation_uri + '> ?end.\n' \
            '}\n' \
            'group by ?start ?end\n' \
            'order by desc(count(*) as ?cnt)\n' \
            'limit ' + str(limit)

    root_results = post_query(query)
    tops = []
    for res in root_results:
        start = res[0][0].text
        end = res[1][0].text
        tops.append({'start': start, 'end': end})
        # tops.append(res[0][0].text)
    return tops


def get_node_name(node_iri):
    return node_iri.split('/')[-1].replace('_', ' ').replace(',', ' ')
