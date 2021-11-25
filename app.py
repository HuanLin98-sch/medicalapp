import os

from transformers import BertTokenizer, BertForMaskedLM, BertModel, GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import torch
# import pandas as pd
from bert_utils import bert_embed_gen
from gpt_utils import generate_gpt_ans
from faiss_utils import Faiss

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from datetime import datetime

app = Flask(__name__)

# we trust longer question less on FAISS, so we will be more likely to use GPT for shorter ques
FAISS_GPT_RATIO = 0.01

QUESTION_BERT_PATH = "resources/all_question_28epoch.pth"
ANSWER_BERT_PATH = "resources/all_answer_28epoch.pth"
BERT_MODEL = 'bert-base-multilingual-uncased'

GPT_PATH = "resources/model_save_3"

EMBEDDINGS_PATH = "resources/lasse_embeddings.pkl"

CORS(app)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")

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
question_model.eval()

# answer model
answer_model = BertForMaskedLM.from_pretrained(BERT_MODEL)
answer_dict = torch.load(
    ANSWER_BERT_PATH, map_location=torch.device('cpu'))
answer_model.load_state_dict(answer_dict)
answer_model.eval()

# GPT model
GPT_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
GPT_model =  GPT2LMHeadModel.from_pretrained(GPT_PATH)
GPT_model.resize_token_embeddings(len(GPT_tokenizer))

# FAISS
faiss_obj = Faiss(EMBEDDINGS_PATH)

@app.route("/get")
# function for the bot response
def get_bot_response():
    question = request.args.get('msg')
    question_length = len(question.split(" "))
    embedding = bert_embed_gen(question, question_model)
    faiss_dist, answer = faiss_obj.get_dist_ans(embedding)
    print(f"FAISS distance: {faiss_dist}")
    # dynamic threshold algo:
    threshold_used = 1.0 - question_length*FAISS_GPT_RATIO
    print(f"Threshold used: {threshold_used}")
    if faiss_dist < threshold_used:
        answer = generate_gpt_ans(question, GPT_tokenizer, GPT_model)
    return answer

if __name__ == '__main__':
    # init()
    app.run(host='localhost', port=5000, debug=True)
