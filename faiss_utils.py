import pickle5 as pickle
import faiss
from faiss import IndexIVFFlat
import numpy as np

class Faiss:
    def __init__(self, embedding_path):
        print("new version")
        with open(embedding_path,'rb') as f:
            self.embeddings = pickle.load(f)
        self.question_embedding = np.array(self.embeddings["question_embedding"])
        self.answer_embedding = np.array(self.embeddings["answer_embedding"])
        self.answers = self.embeddings["answer"]
        self.dimension = self.question_embedding[0].shape[0]
        self.nlist = 5          # number of clusters
        # self.quantiser = faiss.IndexFlatIP(self.dimension) 
        faiss.normalize_L2(self.question_embedding)  
        # print(type(self.quantiser))
        self.index_question = faiss.IndexFlatIP(self.dimension)
        self.index_question.train(self.question_embedding)  # train on the database vectors
        self.index_question.add(self.question_embedding)   # add the vectors and update the index
        # print(self.index_question.is_trained) 
        self.nprobe = 2  # find 2 most similar clusters  
        self.k = 1  # return 3 nearest neighbours
    
    def get_dist_ans(self, question_emb):
        faiss.normalize_L2(question_emb)
        question_distances, question_indices = self.index_question.search(question_emb, self.k)
        # print(question_distances)
        # print(question_indices)
        return question_distances[0][0], self.answers[int(question_indices[0][0])]
