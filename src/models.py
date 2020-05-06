import random
import Levenshtein
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense


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
# 3 & 4. similarity between context and candidate name (common words)
class SupModel:
    def __init__(self):
        self.logistic_reg = LogisticRegression()

    def fit(self, dataset):
        train_set = []
        train_labels = []
        for mention in dataset.mentions:
            context_1 = [x.lower().strip() for x in mention.contexts[0]]
            context_2 = [x.lower().strip() for x in mention.contexts[1]]
            for candidate in mention.candidates:
                words = candidate.name.lower().split()
                context_sim_1 = sum([1 if x.strip() in context_1 else 0 for x in words])
                context_sim_2 = sum([1 if x.strip() in context_2 else 0 for x in words])
                train_set.append([candidate.prob, math.exp(-Levenshtein.distance(candidate.name, mention.surface)),
                                  context_sim_1 / len(words), context_sim_2 / len(words)])
                train_labels.append(1 if mention.gt.id == candidate.id else 0)
        train_set = np.array(train_set)
        train_labels = np.array(train_labels)

        self.logistic_reg.fit(train_set, train_labels)

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            dev_set = []
            context_1 = [x.lower().strip() for x in mention.contexts[0]]
            context_2 = [x.lower().strip() for x in mention.contexts[1]]
            for candidate in mention.candidates:
                words = candidate.name.lower().split()
                context_sim_1 = sum([1 if x.strip() in context_1 else 0 for x in words])
                context_sim_2 = sum([1 if x.strip() in context_2 else 0 for x in words])
                dev_set.append([candidate.prob, math.exp(-Levenshtein.distance(candidate.name, mention.surface)),
                                context_sim_1 / len(words), context_sim_2 / len(words)])
            dev_set = np.array(dev_set)
            if mention.candidates:
                pred = self.logistic_reg.predict_proba(dev_set)
                pred_cids.append(mention.candidates[np.argmax(pred[:, 1])].id)
            else:
                pred_cids.append('NIL')
        return pred_cids


# 1. prior probability (1 D)
# 2. distance between surface name and candidate name (1 D)
# 3. similarity between context and candidate name (2 D)
# 4. candidate entity embedding (300 D)
# 5. sum of context word embeddings (300 D)
# 6. cosine similarity between the two embedding (1 D)
class SupModelWithEmbedding:
    def __init__(self):
        self.logistic_reg = LogisticRegression(max_iter=200)

        with open('../data/embeddings/ent2embed.pk', 'rb') as rf:
            self.ent2embed = pickle.load(rf)
        with open('../data/embeddings/word2embed.pk', 'rb') as rf:
            self.word2embed = pickle.load(rf)

    def fit(self, dataset):
        train_set = []
        train_labels = []
        for mention in dataset.mentions:
            context_1 = [x.lower().strip() for x in mention.contexts[0]]
            context_2 = [x.lower().strip() for x in mention.contexts[1]]
            word_emb = np.zeros((300,))
            for word in mention.contexts[0]:
                word_emb += self.word2embed.get(word, np.zeros(300,))
            for word in mention.contexts[1]:
                word_emb += self.word2embed.get(word, np.zeros(300,))

            for candidate in mention.candidates:
                feat = [candidate.prob,
                        math.exp(-Levenshtein.distance(candidate.name, mention.surface))]
                words = candidate.name.lower().split()
                context_sim_1 = sum([1 if x.strip() in context_1 else 0 for x in words])
                context_sim_2 = sum([1 if x.strip() in context_2 else 0 for x in words])
                feat.append(context_sim_1 / len(words))
                feat.append(context_sim_2 / len(words))
                ent = '_'.join(candidate.name.split(' '))
                ent_emb = self.ent2embed[ent]
                feat.extend(ent_emb)
                feat.extend(word_emb)
                feat.append(cosine_sim_vectors(word_emb, ent_emb))

                train_set.append(feat)
                train_labels.append(1 if mention.gt.id == candidate.id else 0)

        train_set = np.array(train_set)
        train_labels = np.array(train_labels)

        self.logistic_reg.fit(train_set, train_labels)

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            context_1 = [x.lower().strip() for x in mention.contexts[0]]
            context_2 = [x.lower().strip() for x in mention.contexts[1]]
            word_emb = np.zeros(300,)
            for word in mention.contexts[0]:
                word_emb += self.word2embed.get(word, np.zeros(300,))
            for word in mention.contexts[1]:
                word_emb += self.word2embed.get(word, np.zeros(300,))

            dev_set = []
            for candidate in mention.candidates:
                feat = [candidate.prob,
                        math.exp(-Levenshtein.distance(candidate.name, mention.surface))]
                words = candidate.name.lower().split()
                context_sim_1 = sum([1 if x.strip() in context_1 else 0 for x in words])
                context_sim_2 = sum([1 if x.strip() in context_2 else 0 for x in words])
                feat.append(context_sim_1 / len(words))
                feat.append(context_sim_2 / len(words))
                ent = '_'.join(candidate.name.split(' '))
                ent_emb = self.ent2embed.get(ent, np.zeros(300,))
                feat.extend(ent_emb)
                feat.extend(word_emb)
                feat.append(cosine_sim_vectors(word_emb, ent_emb))

                dev_set.append(feat)

            dev_set = np.array(dev_set)
            if mention.candidates:
                pred = self.logistic_reg.predict_proba(dev_set)
                pred_cids.append(mention.candidates[np.argmax(pred[:, 1])].id)
            else:
                pred_cids.append('NIL')
        return pred_cids


class SupModelNN:
    def __init__(self):
        self.net = Sequential()
        self.net.add(Dense(16, activation='relu'))
        self.net.add(Dense(4, activation='relu'))
        self.net.add(Dense(1, activation='sigmoid'))
        self.net.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        with open('../data/embeddings/ent2embed.pk', 'rb') as rf:
            self.ent2embed = pickle.load(rf)
        with open('../data/embeddings/word2embed.pk', 'rb') as rf:
            self.word2embed = pickle.load(rf)

    def fit(self, dataset):
        train_set = []
        train_labels = []
        for mention in dataset.mentions:
            context_1 = [x.lower().strip() for x in mention.contexts[0]]
            context_2 = [x.lower().strip() for x in mention.contexts[1]]
            word_emb = np.zeros((300,))
            for word in mention.contexts[0]:
                word_emb += self.word2embed.get(word, np.zeros(300,))
            for word in mention.contexts[1]:
                word_emb += self.word2embed.get(word, np.zeros(300,))

            for candidate in mention.candidates:
                feat = [candidate.prob,
                        math.exp(-Levenshtein.distance(candidate.name, mention.surface))]
                words = candidate.name.lower().split()
                context_sim_1 = sum([1 if x.strip() in context_1 else 0 for x in words])
                context_sim_2 = sum([1 if x.strip() in context_2 else 0 for x in words])
                feat.append(context_sim_1 / len(words))
                feat.append(context_sim_2 / len(words))
                ent = '_'.join(candidate.name.split(' '))
                ent_emb = self.ent2embed[ent]
                feat.extend(ent_emb)
                feat.extend(word_emb)
                feat.append(cosine_sim_vectors(word_emb, ent_emb))

                train_set.append(feat)
                train_labels.append(1 if mention.gt.id == candidate.id else 0)

        train_set = np.array(train_set)
        train_labels = np.array(train_labels)

        self.net.fit(train_set, train_labels, epochs=10, batch_size=100, verbose=False)

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            context_1 = [x.lower().strip() for x in mention.contexts[0]]
            context_2 = [x.lower().strip() for x in mention.contexts[1]]
            word_emb = np.zeros(300,)
            for word in mention.contexts[0]:
                word_emb += self.word2embed.get(word, np.zeros(300,))
            for word in mention.contexts[1]:
                word_emb += self.word2embed.get(word, np.zeros(300,))

            dev_set = []
            for candidate in mention.candidates:
                feat = [candidate.prob,
                        math.exp(-Levenshtein.distance(candidate.name, mention.surface))]
                words = candidate.name.lower().split()
                context_sim_1 = sum([1 if x.strip() in context_1 else 0 for x in words])
                context_sim_2 = sum([1 if x.strip() in context_2 else 0 for x in words])
                feat.append(context_sim_1 / len(words))
                feat.append(context_sim_2 / len(words))
                ent = '_'.join(candidate.name.split(' '))
                ent_emb = self.ent2embed.get(ent, np.zeros(300,))
                feat.extend(ent_emb)
                feat.extend(word_emb)
                feat.append(cosine_sim_vectors(word_emb, ent_emb))

                dev_set.append(feat)

            dev_set = np.array(dev_set)
            if mention.candidates:
                pred = self.net.predict_proba(dev_set)
                pred_cids.append(mention.candidates[np.argmax(pred)].id)
            else:
                pred_cids.append('NIL')
        return pred_cids


# 1. prior probability (1 D)
# 2. distance between surface name and candidate name (1 D)
# 3. similarity between context and candidate name (2 D)
# 4. candidate entity embedding (300 D)
# 5. sum of context word embeddings (300 D)
# 6. cosine similarity between the two embedding (1 D)
# 7. maximum similarity between context and candidate's documents in knowledge base (2 D)
class MyModel:
    def __init__(self):
        self.logistic_reg = LogisticRegression(max_iter=200)

        with open('../data/embeddings/ent2embed.pk', 'rb') as rf:
            self.ent2embed = pickle.load(rf)
        with open('../data/embeddings/word2embed.pk', 'rb') as rf:
            self.word2embed = pickle.load(rf)

    def fit(self, dataset, knowledge_base):
        train_set = []
        train_labels = []
        for mention in dataset.mentions:
            context_1 = [x.lower().strip() for x in mention.contexts[0]]
            context_2 = [x.lower().strip() for x in mention.contexts[1]]

            word_emb = np.zeros((300,))
            for word in mention.contexts[0]:
                word_emb += self.word2embed.get(word, np.zeros(300,))
            for word in mention.contexts[1]:
                word_emb += self.word2embed.get(word, np.zeros(300,))

            for candidate in mention.candidates:
                feat = [candidate.prob,
                        math.exp(-Levenshtein.distance(candidate.name, mention.surface))]

                words = candidate.name.lower().split()
                context_sim_1 = sum([1 if x.strip() in context_1 else 0 for x in words])
                context_sim_2 = sum([1 if x.strip() in context_2 else 0 for x in words])
                feat.append(context_sim_1 / len(words))
                feat.append(context_sim_2 / len(words))

                ent = '_'.join(candidate.name.split(' '))
                ent_emb = self.ent2embed[ent]
                feat.extend(ent_emb)
                feat.extend(word_emb)
                feat.append(cosine_sim_vectors(word_emb, ent_emb))

                max_sim_1 = 0
                max_sim_2 = 0
                if candidate.id in knowledge_base.documents:
                    for section in knowledge_base.documents[candidate.id].sections:
                        for sentence in section:
                            s = sentence.lower()
                            tmp_sim_1 = sum([1 if x.strip() in s else 0 for x in context_1])
                            tmp_sim_2 = sum([1 if x.strip() in s else 0 for x in context_2])
                            if tmp_sim_1 > max_sim_1:
                                max_sim_1 = tmp_sim_1
                            if tmp_sim_2 > max_sim_2:
                                max_sim_2 = tmp_sim_2
                feat.append(max_sim_1 / len(context_1) if len(context_1) > 0 else 0)
                feat.append(max_sim_2 / len(context_2) if len(context_2) > 0 else 0)

                train_set.append(feat)
                train_labels.append(1 if mention.gt.id == candidate.id else 0)

        train_set = np.array(train_set)
        train_labels = np.array(train_labels)

        self.logistic_reg.fit(train_set, train_labels)

    def predict(self, dataset, knowledge_base):
        pred_cids = []
        for mention in dataset.mentions:
            context_1 = [x.lower().strip() for x in mention.contexts[0]]
            context_2 = [x.lower().strip() for x in mention.contexts[1]]

            word_emb = np.zeros(300,)
            for word in mention.contexts[0]:
                word_emb += self.word2embed.get(word, np.zeros(300,))
            for word in mention.contexts[1]:
                word_emb += self.word2embed.get(word, np.zeros(300,))

            dev_set = []
            for candidate in mention.candidates:
                feat = [candidate.prob,
                        math.exp(-Levenshtein.distance(candidate.name, mention.surface))]

                words = candidate.name.lower().split()
                context_sim_1 = sum([1 if x.strip() in context_1 else 0 for x in words])
                context_sim_2 = sum([1 if x.strip() in context_2 else 0 for x in words])
                feat.append(context_sim_1 / len(words))
                feat.append(context_sim_2 / len(words))

                ent = '_'.join(candidate.name.split(' '))
                ent_emb = self.ent2embed.get(ent, np.zeros(300,))
                feat.extend(ent_emb)
                feat.extend(word_emb)
                feat.append(cosine_sim_vectors(word_emb, ent_emb))

                max_sim_1 = 0
                max_sim_2 = 0
                if candidate.id in knowledge_base.documents:
                    for section in knowledge_base.documents[candidate.id].sections:
                        for sentence in section:
                            s = sentence.lower()
                            tmp_sim_1 = sum([1 if x.strip() in s else 0 for x in context_1])
                            tmp_sim_2 = sum([1 if x.strip() in s else 0 for x in context_2])
                            if tmp_sim_1 > max_sim_1:
                                max_sim_1 = tmp_sim_1
                            if tmp_sim_2 > max_sim_2:
                                max_sim_2 = tmp_sim_2
                feat.append(max_sim_1 / len(context_1) if len(context_1) > 0 else 0)
                feat.append(max_sim_2 / len(context_2) if len(context_2) > 0 else 0)

                dev_set.append(feat)

            dev_set = np.array(dev_set)
            if mention.candidates:
                pred = self.logistic_reg.predict_proba(dev_set)
                pred_cids.append(mention.candidates[np.argmax(pred[:, 1])].id)
            else:
                pred_cids.append('NIL')
        return pred_cids


class MyModelNN:
    def __init__(self):
        self.net = Sequential()
        self.net.add(Dense(16, activation='relu'))
        self.net.add(Dense(4, activation='relu'))
        self.net.add(Dense(1, activation='sigmoid'))
        self.net.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        with open('../data/embeddings/ent2embed.pk', 'rb') as rf:
            self.ent2embed = pickle.load(rf)
        with open('../data/embeddings/word2embed.pk', 'rb') as rf:
            self.word2embed = pickle.load(rf)

    def fit(self, dataset, knowledge_base):
        train_set = []
        train_labels = []
        for mention in dataset.mentions:
            context_1 = [x.lower().strip() for x in mention.contexts[0]]
            context_2 = [x.lower().strip() for x in mention.contexts[1]]

            word_emb = np.zeros((300,))
            for word in mention.contexts[0]:
                word_emb += self.word2embed.get(word, np.zeros(300,))
            for word in mention.contexts[1]:
                word_emb += self.word2embed.get(word, np.zeros(300,))

            for candidate in mention.candidates:
                feat = [candidate.prob,
                        math.exp(-Levenshtein.distance(candidate.name, mention.surface))]

                words = candidate.name.lower().split()
                context_sim_1 = sum([1 if x.strip() in context_1 else 0 for x in words])
                context_sim_2 = sum([1 if x.strip() in context_2 else 0 for x in words])
                feat.append(context_sim_1 / len(words))
                feat.append(context_sim_2 / len(words))

                ent = '_'.join(candidate.name.split(' '))
                ent_emb = self.ent2embed[ent]
                feat.extend(ent_emb)
                feat.extend(word_emb)
                feat.append(cosine_sim_vectors(word_emb, ent_emb))

                max_sim_1 = 0
                max_sim_2 = 0
                if candidate.id in knowledge_base.documents:
                    for section in knowledge_base.documents[candidate.id].sections:
                        for sentence in section:
                            s = sentence.lower()
                            tmp_sim_1 = sum([1 if x.strip() in s else 0 for x in context_1])
                            tmp_sim_2 = sum([1 if x.strip() in s else 0 for x in context_2])
                            if tmp_sim_1 > max_sim_1:
                                max_sim_1 = tmp_sim_1
                            if tmp_sim_2 > max_sim_2:
                                max_sim_2 = tmp_sim_2
                feat.append(max_sim_1 / len(context_1) if len(context_1) > 0 else 0)
                feat.append(max_sim_2 / len(context_2) if len(context_2) > 0 else 0)

                train_set.append(feat)
                train_labels.append(1 if mention.gt.id == candidate.id else 0)

        train_set = np.array(train_set)
        train_labels = np.array(train_labels)

        self.net.fit(train_set, train_labels, epochs=10, batch_size=100, verbose=False)

    def predict(self, dataset, knowledge_base):
        pred_cids = []
        for mention in dataset.mentions:
            context_1 = [x.lower().strip() for x in mention.contexts[0]]
            context_2 = [x.lower().strip() for x in mention.contexts[1]]

            word_emb = np.zeros(300,)
            for word in mention.contexts[0]:
                word_emb += self.word2embed.get(word, np.zeros(300,))
            for word in mention.contexts[1]:
                word_emb += self.word2embed.get(word, np.zeros(300,))

            dev_set = []
            for candidate in mention.candidates:
                feat = [candidate.prob,
                        math.exp(-Levenshtein.distance(candidate.name, mention.surface))]

                words = candidate.name.lower().split()
                context_sim_1 = sum([1 if x.strip() in context_1 else 0 for x in words])
                context_sim_2 = sum([1 if x.strip() in context_2 else 0 for x in words])
                feat.append(context_sim_1 / len(words))
                feat.append(context_sim_2 / len(words))

                ent = '_'.join(candidate.name.split(' '))
                ent_emb = self.ent2embed.get(ent, np.zeros(300,))
                feat.extend(ent_emb)
                feat.extend(word_emb)
                feat.append(cosine_sim_vectors(word_emb, ent_emb))

                max_sim_1 = 0
                max_sim_2 = 0
                if candidate.id in knowledge_base.documents:
                    for section in knowledge_base.documents[candidate.id].sections:
                        for sentence in section:
                            s = sentence.lower()
                            tmp_sim_1 = sum([1 if x.strip() in s else 0 for x in context_1])
                            tmp_sim_2 = sum([1 if x.strip() in s else 0 for x in context_2])
                            if tmp_sim_1 > max_sim_1:
                                max_sim_1 = tmp_sim_1
                            if tmp_sim_2 > max_sim_2:
                                max_sim_2 = tmp_sim_2
                feat.append(max_sim_1 / len(context_1) if len(context_1) > 0 else 0)
                feat.append(max_sim_2 / len(context_2) if len(context_2) > 0 else 0)

                dev_set.append(feat)

            dev_set = np.array(dev_set)
            if mention.candidates:
                pred = self.net.predict_proba(dev_set)
                pred_cids.append(mention.candidates[np.argmax(pred)].id)
            else:
                pred_cids.append('NIL')
        return pred_cids
