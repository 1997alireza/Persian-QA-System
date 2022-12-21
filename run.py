from src.processing.answer_generator import get_answer_generator
from src.web_interface.interface import start


if __name__ == '__main__':
    mode = 'web'  # 'web' or 'command'

    if mode is 'web':
        start()

    elif mode is 'command':
        answer_generator = get_answer_generator(log=True)
        while True:
            input_sentence = input('Enter a sentence:')
            if input_sentence == 'q':
                exit()
            info = answer_generator(input_sentence, check_all_question_entities=True, multiple_relations=True,
                                    return_intermediate_info=True)
            print(info, '\n')

