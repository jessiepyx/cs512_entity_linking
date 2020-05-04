import random
import Levenshtein
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

stopwords = stopwords.words('english')


def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text


def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


class RandomModel:
    def __init__(self):
        pass

    def fit(self, dataset):
        # fill this function if your model requires training
        pass

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            pred_cids.append(random.choice(mention.candidates).id if mention.candidates else 'NIL')
        return pred_cids


class PriorModel:
    def __init__(self):
        pass

    def fit(self, dataset):
        pass

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            pred_cids.append(max(mention.candidates, key=lambda x: x.prob).id if mention.candidates else 'NIL')
        return pred_cids


# 1. prior probability
# 2. distance between surface name and candidate name (Levenshtein distance)
# 3 & 4. similarity between context and candidate name (cosine similarity)
class SupModel:
    def __init__(self):
        super(SupModel, self).__init__()
        self.lrate = 0.01
        self.in_size = 4
        self.out_size = 2

        self.features = nn.Linear(self.in_size, self.out_size)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.features.parameters(), self.lrate)
        self.classifier = nn.Softmax()

    def get_parameters(self):
        return self.features.parameters()

    def forward(self, x):
        return self.features(x)

    def step(self, x, y):
        self.optimizer.zero_grad()

        # forward
        y_hat = self.forward(x)

        # loss
        loss = self.criterion(y_hat, y)
        L = loss.item()

        # backward
        loss.backward()
        self.optimizer.step()

        return L

    def fit(self, dataset):
        # make tensors
        train_set = []
        train_labels = []
        for mention in dataset.mentions:
            idx = 1
            sentences = [mention.contexts[0], mention.contexts[1]]
            for candidate in mention.candidates:
                sentences.append(candidate.name)
            cleaned = list(map(clean_string, sentences))
            vectorizer = CountVectorizer().fit_transform(cleaned)
            vectors = vectorizer.toarray()
            for candidate in mention.candidates:
                idx += 1
                train_set.append([candidate.prob, math.exp(-Levenshtein.distance(candidate.name, mention.surface)),
                                  cosine_sim_vectors(vectors[0], vectors[idx]),
                                  cosine_sim_vectors(vectors[1], vectors[idx])])
                train_labels.append(1 if mention.gt.id == candidate.id else 0)
        train_set = torch.tensor(train_set, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.int64)

        # train
        n_iter = 2000
        batch_size = 200
        N = train_set.shape[0]
        for batch in range(n_iter):
            # get batch data
            idx = torch.randperm(N)
            x_batch = train_set[idx[:batch_size]]
            y_batch = train_labels[idx[:batch_size]]
            loss = self.step(x_batch, y_batch)
            print(loss)

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            idx = 1
            sentences = [mention.contexts[0], mention.contexts[1]]
            for candidate in mention.candidates:
                sentences.append(candidate.name)
            cleaned = list(map(clean_string, sentences))
            vectorizer = CountVectorizer().fit_transform(cleaned)
            vectors = vectorizer.toarray()
            dev_set = []
            for candidate in mention.candidates:
                idx += 1
                dev_set.append([candidate.prob, math.exp(-Levenshtein.distance(candidate.name, mention.surface)),
                                cosine_sim_vectors(vectors[0], vectors[idx]),
                                cosine_sim_vectors(vectors[1], vectors[idx])])
            dev_set = torch.tensor(dev_set, dtype=torch.float32)
            pred_cids.append(mention.candidates[np.argmax(
                self.forward(dev_set).detach().numpy()[:, 1])].id if mention.candidates else 'NIL')
        return pred_cids
