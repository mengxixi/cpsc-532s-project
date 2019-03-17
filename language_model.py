import numpy as np

class GloVe():
    def __init__(self, path, dim):
        self.dim = dim
        self.word_embedding_dict = {}
        with open(path) as f:
            for line in f:
                values = line.split()
                embedding = values[-dim:]
                word = ''.join(values[:-dim])
                self.word_embedding_dict[word] = np.asarray(embedding, dtype=np.float32)
    
    def get_word_vector(self, word):
        if word not in self.word_embedding_dict.keys():
            embedding = np.random.uniform(low=-1, high=1, size=self.dim).astype(np.float32)
            self.word_embedding_dict[word] = embedding
            return embedding
        else:
            return self.word_embedding_dict[word]