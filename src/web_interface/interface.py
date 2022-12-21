from flask import Flask, render_template, request
import json
from src.processing.answer_generator import get_answer_generator


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/answer', methods=['POST'])
def get_answer():
    print('a request on /answer is received')
    question = request.form['question']
    multiple_relations = request.form['multiple_relations'] == 'true'

    is_admin = False
    if 'authentication_id' in request.form and 'authentication_code' in request.form:
        authentication_id = request.form['authentication_id']
        authentication_code = request.form['authentication_code']

        is_admin = authentication_id == 'admin' and authentication_code == '0000'
        # TODO: authentication_code must be checked with the correct value which is sent to the client

    info = answer_generator(question,
                            check_all_question_entities=True, multiple_relations=multiple_relations,
                            return_intermediate_info=is_admin)

    return json.dumps(info)


def start():
    global app, answer_generator
    answer_generator = get_answer_generator(log=True)
    app.run()


if __name__ == "__main__":
    start()
