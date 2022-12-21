import pandas as pd
from src.processing.answer_generator import get_answer_generator
import paths


def evaluation(test_dataset_adr=paths.dataset+'end2end_test_dataset.csv'):
    ds = pd.read_csv(test_dataset_adr)
    answer_generator = get_answer_generator()

    # wrong_f = open('../logs/end2end_evaluation wrong_answers.txt', 'a+', encoding='utf8')
    # wrong_f.write('<question>, <query>, <relation>\n')

    correct_answers = 0

    try:
        for i in range(0, len(ds)):
            # if i % 50 == 0:
            print(i, '/', len(ds))

            row = ds.iloc[i, :]
            question = row[0]
            answer_entity = row[1]

            info = answer_generator(question, return_intermediate_info=True)
            info = info['ret_ans']

            relation_label = info['rel']
            try:
                first_result = info['result'][0]
                answers = first_result['answers']
                query = first_result['query']
            except Exception:
                answers, query = None, None

            if answers is not None and answer_entity in answers:
                correct_answers += 1
            # else:
            #     wrong_f.write((','.join([question, str(query), relation_label])).replace('\n', '') + '\n')


    finally:
        # wrong_f.close()
        print('accuracy: {} ({}/{})'.format(correct_answers/len(ds), correct_answers, len(ds)))


if __name__ == '__main__':
    evaluation()
