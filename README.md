# Dr Robot

Developed by: Team 7 - Edwin, Huan Lin, Wei Jie, Zhi An

## Introduction
Dr Robot is a general medical conversational chatbot developed by Team 7 as part of our CS425 final project. It is a hybrid chatbot that combines both retrieval-based (Finetuned BERT + FAISS Search) answers with generative-based (Finetuned GPT2) answers using a dynamic thresholding via our own decision module.

## Quick Start
1. Open a terminal session in the root folder
2. ```flask run```
3. The chatbot should now be accessible at the localhost :)
4. Begin Chatting!

## Chatbot Implementation
Resources for running the model can be found at the following Google Drive link. This includes the saved model checkpoints for Multilingual BERT (Question & Answer), the embeddings of the training set and the finetuned GPT2 model (FAISS-mBert-Lasse) 
https://drive.google.com/drive/folders/1GEcx7XywfUmGmztXb3ljI8RkLeo90bN1?usp=sharing
*Please use an SMU account to access*

## Training Details
For an in-depth explaination of our training process, please refer to our report.

### Translation & Data Preparation

### BERT Finetuning

### FAISS Similarity Search
FAISS Similarity Search is used to create more Question Answer pair embeddings to be used for GPT training. It is based on the FAISS library. The codes for create more Question Answer pair embeddings can be found at FAISS_Similarity_Search folder. 

There are three .ipynb files and each of them are used for different embeddings due to the format of its dictionary construct. Lasse dataset's embedding should be used with *FAISS_Similarity_Search/FAISS_Similarity_Search_Lasse.ipynb*. MedDiaglog Chinese dataset's embedding should be used with *FAISS_Similarity_Search/FAISS_Similarity_Search_MedDiaglog_Chinese.ipynb* For using both datasets together, it should be used with *FAISS_Similarity_Search/FAISS_Similarity_Search_BOTH.ipynb*.

### GPT Training
GPT2-small was used as the base model for finetuning. Most of the implementation is largely based on finetuning code provided by the *Huggingface* library. The code for finetuning can be found at *gpt_training/gpt_training.py*

To generate answers using a finetuned model on a test set of questions, please refer to *gpt_training/test_pipeline_runner_gpt.py*

### Evaluation
There were 2 test datasets used for evaluation - MedDialog-EN and Lasse Perturbed. These can both be found under *evaluation*. 

Calcuation of BLUERT-20 scores can be done using the script *evaluation/bluert20.py*


