# Dr Robot

Developed by: Team 7 - Edwin, Huan Lin, Wei Jie, Zhi An

## Introduction
Dr Robot is a general medical conversational chatbot developed by Team 7 as part of our CS425 final project. It is a hybrid chatbot that combines both retrieval-based (Finetuned BERT + FAISS Search) answers with generative-based (Finetuned GPT2) answers using a dynamic thresholding via our own decision module.

## Quick Start
1. Open a terminal session in the root folder
2. Download and save the required resources to run the application using the instructions below
3. ```flask run```
4. The chatbot should now be accessible at the localhost :)
5. Begin Chatting!

## Chatbot Implementation
Resources for running the model can be found at the following Google Drive link. This includes the saved model checkpoints for Multilingual BERT (Question & Answer), the embeddings of the training set and the finetuned GPT2 model (FAISS-mBert-Lasse) which are all used in the hybrid model. 
https://drive.google.com/drive/folders/1GEcx7XywfUmGmztXb3ljI8RkLeo90bN1?usp=sharing
*Please use an SMU account to access*

You can simply download all the files in the gdrive above and create a folder called `resources` in the root directory and paste those files in.

## Training Details
For an in-depth explanation of our training process, please refer to our report.

### Translation & Data Preparation

Our team used the dataset provided by [MedDialog](https://drive.google.com/drive/folders/11sglwm6-cY7gjeqlZaMxL_MDKDMLdhym). We created a Python script `translate.py` that reads and translates the texts to English using the `opus-mt` model from [EasyNMT](https://github.com/UKPLab/EasyNMT). We chose this model because it had the fastest translation speed at a decent translation accuracy as compared to the other models. The models were evaluated using `evaluate.py` on a [dictionary](https://github.com/HikaruSama233/Chinese-English-dictionary-of-medical-and-health) of Chinese-English medical terms crawled from CNKI.

### BERT Finetuning
Different pre-trained BERT models can be used to fine-tune by specifying in the `PRETRAINED_NAME` variable in the training files in the `BERT fine-tuning` directory. The one that is in the code uses the PubMedBERT from hugging face. It also uses the lasses dataset for it's finetuning, a sample Lasse dataset has also been rpovided in the directory. to run this code, a GPU is required since we are using the Trainer class privided by the Transformers library. 

Answers and question BERT encoders are also fine-tuned separately using the *BERT fine-tuning/answer_bert_training.py* and *BERT fine-tuning/question_bert_training.py* files respectively.

*bert_utils.py* contains the script to embed a given input sentence into embeddings which will then be used in the FAISS similarity search and hybrid model.

### FAISS Similarity Search
FAISS Similarity Search is used to create more Question Answer pair embeddings to be used for GPT training. It is based on the FAISS library. The codes for create more Question Answer pair embeddings can be found at FAISS_Similarity_Search folder. 

There are three .ipynb files and each of them are used for different embeddings due to the format of its dictionary construct. 

Lasse dataset's embedding should be used with *FAISS_Similarity_Search/FAISS_Similarity_Search_Lasse.ipynb*.

MedDiaglog Chinese dataset's embedding should be used with *FAISS_Similarity_Search/FAISS_Similarity_Search_MedDiaglog_Chinese.ipynb*

To use both datasets' embeddings together, it should be used with *FAISS_Similarity_Search/FAISS_Similarity_Search_BOTH.ipynb*.

### GPT Training
GPT2-small was used as the base model for finetuning. Most of the implementation is largely based on finetuning code provided by the *Huggingface* library. The code for finetuning can be found at *gpt_training/gpt_training.py*

To generate answers using a finetuned model on a test set of questions, please refer to *gpt_training/test_pipeline_runner_gpt.py*

### Evaluation
There were 2 test datasets used for evaluation - MedDialog-EN and Lasse Perturbed. These can both be found under *evaluation*. 

Calculation of BLUERT-20 scores can be done using the script *evaluation/bluert20.py*

