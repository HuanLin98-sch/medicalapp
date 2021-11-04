import pickle5 as pickle
import faiss
from faiss import IndexIVFFlat
import numpy as np

class Faiss:
    def __init__(self, embedding_path):
        with open(embedding_path,'rb') as f:
            self.embeddings = pickle.load(f)
        self.question_embedding = np.array(self.embeddings["question_embedding"])
        self.answer_embedding = np.array(self.embeddings["answer_embedding"])
        self.answers = self.embeddings["answer"]
        self.dimension = 768
        self.nlist = 5          # number of clusters
        self.quantiser = faiss.IndexFlatL2(self.dimension)  
        print(type(self.quantiser))
        self.index_question = IndexIVFFlat(self.quantiser, self.dimension, self.nlist)
        self.index_question.train(self.question_embedding)  # train on the database vectors
        self.index_question.add(self.question_embedding)   # add the vectors and update the index
        print(self.index_question.is_trained) 
        self.nprobe = 2  # find 2 most similar clusters  
        self.k = 1  # return 3 nearest neighbours
    
    def get_dist_ans(self, question_emb):
        question_distances, question_indices = self.index_question.search(question_emb, self.k)
        print(question_distances)
        print(question_indices)
        return question_distances[0][0], self.answers[int(question_indices[0][0])]
