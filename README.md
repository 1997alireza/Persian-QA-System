Persian Question Answering System
====================

The purpose of the project is to build a model that can generate answers to simple questions based on the facts in a knowledge-base. Simple questions are are the ones that can be answered using just one relation. The answer is extracted using SPARQL queries, by knowing the start node and the type of the relation. The start node is the named-entity in the question, which is identified using a named-entity recognition made by a neural network. A classifier is necessary to figure out the relation in the question. SVM and CNN classification models are used to classify the relations. To use these models, sentences should be mapped to fixed-size vectors. For this mapping, Bag of Words, TF-IDF and Word2Vec models are used.

## Bachelor Thesis

This project is contributed to my bachelor thesis which is available [here](docs/Thesis.pdf).

## Installation

Install required packages:
    
    pip install -r requirements.txt
    
If you need to use [the evaluation part](src/evaluation/), install their required packages too:

    pip install -r requirements-eval.txt

## Usage

### Web Interface

<p align="center" ><img src="docs/Thesis%20Latex/figures/interface/admin-double-multiple-new.png" data-canonical-src="docs/Thesis%20Latex/figures/interface/admin-double-multiple-new.png" width="90%" /> </p>

Run
    
    python3 run.py
    
### APIs

Three APIs are provided to use this system which are explained below. You can access them through `APIs.py`.

1. At first, you should provide a CSV file containing the pairs of relations and question templates:
    
    | label(relation) 	| sentence 	|
    | :-: | :-: |
    | پایتخت یک کشور | پایتخت کشور ررر چه نام دارد |
    | ناشر کتاب | انتشاراتی که اثر ککک را منتشر کرده است کیست |

    * If you want to add a new relation, edit [rel_name2uri.csv](src/master/dataset/rel_name2uri.csv) file. Use three identical characters with zero to three extra characters as start/end entity keys.
    
2. Using the following function you can generate the final dataset, which is required for the training process, from the relation and question template pairs file:

    ```
    generate_final_dataset(rel_question_source, append=False)
    ``` 

    * If you pass `append` as True, previous data will not be erased.

3. Call the following function to train the models:

    ```
    train()
    ``` 

4. Now you can use the system through the next function providing a generator which takes a question and returns an answer:

    ```
    run(multiple_relations=False)
    ``` 
    
    * If `multiple_relations` is True, all of the relations with high probabilities is checked instead of just the best one.
