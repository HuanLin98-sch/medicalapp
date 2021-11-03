import os

from transformers import BertTokenizer, BertForMaskedLM, BertModel
import torch
import pandas as pd
from bert_utils import bert_embed_gen

from flask import Flask, request, jsonify
from flask_cors import CORS

from datetime import datetime

app = Flask(__name__)

QUESTION_BERT_PATH = "resources/all_question_28epoch.pth"
ANSWER_BERT_PATH = "resources/all_answer_28epoch.pth"
BERT_MODEL = 'bert-base-multilingual-uncased'

CORS(app)

@app.route("/health")
def health_check():
    return jsonify(
        {
            "message": "Service is healthy."
        }
    ), 200

print('LOADING RESOURCES')
# question model
question_model = BertForMaskedLM.from_pretrained(BERT_MODEL)
question_dict = torch.load(
    QUESTION_BERT_PATH, map_location=torch.device('cpu'))
question_model.load_state_dict(question_dict)

# answer model
answer_model = BertForMaskedLM.from_pretrained(BERT_MODEL)
answer_dict = torch.load(
    ANSWER_BERT_PATH, map_location=torch.device('cpu'))
answer_model.load_state_dict(answer_dict)

@app.route("/question", methods=['POST'])
def question_asked():
    try:
        question = request.json.get('question')
        embedding = bert_embed_gen(question, question_model)
        

    except Exception as e:
        return jsonify(
            {
                "message": "An error occurred while processing the question.",
                "error": str(e)
            }
        ), 500

    return jsonify(
        {
            "answer": str(embedding)
        }
    ), 201


if __name__ == '__main__':
    # init()
    app.run(host='localhost', port=5000, debug=True)
