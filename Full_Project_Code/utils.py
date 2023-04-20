# Import statements for the functions
import re
import string
import random
import pickle
import numpy as np
import pandas as pd
from itertools import chain
from itertools import product
from scipy import spatial
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

# Calculates cosine similarity between two lists of tags
def get_similarity_embedding(source_tags, target_tags):
    similarity = 1 - spatial.distance.cosine(source_tags, target_tags)
    return similarity

# Gets the tags and their frequencies for domains in the Books dataset
def gather_tags_books(df):
    categories = df['tag'].values
    tags_punc = list(categories)
    pattern = "[0123456789" + re.escape(string.punctuation) + "]"
    tags_clean = list(map(lambda x: re.sub(pattern, '', str(x).lower()), tags_punc))
    tags_split = list(map(lambda x: x.split(), tags_clean))
    tags_words = list(chain.from_iterable(tags_split))
    tags_words = list(tags_words)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in tags_words if word not in stop_words])
    tags = list(text.split(' '))
    d = {}
    for i in tags:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    tags = list(set(tags))
    return [tags, d]

# Gets the tags and their frequencies for domains in the Amazon dataset
def gather_tags_amazon(df):
    categories = df['category'].values
    flatten_list = list(chain.from_iterable(categories))
    tags_punc = list(flatten_list)
    pattern = "[0123456789" + re.escape(string.punctuation) + "]"
    tags_clean = list(map(lambda x: re.sub(pattern, '', x.lower()), tags_punc))
    tags_split = list(map(lambda x: x.split(), tags_clean))
    tags_words = list(chain.from_iterable(tags_split))
    tags_words = list(tags_words)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in tags_words if word.lower() not in stop_words])
    tags = list(text.split(' '))
    d = {}
    for i in tags:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
            
    tags = list(set(tags))
    return [tags, d]

# Gets the tags and their frequencies for domains in the Movielens dataset
def gather_tags_movie(df):
    categories = df['tag'].values
    tags_punc = list(categories)
    pattern = "[0123456789" + re.escape(string.punctuation) + "]"
    tags_clean = list(map(lambda x: re.sub(pattern, '', str(x).lower()), tags_punc))
    tags_split = list(map(lambda x: x.split(), tags_clean))
    tags_words = list(chain.from_iterable(tags_split))
    tags_words = list(tags_words)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in tags_words if word not in stop_words])
    tags = list(text.split(' '))
    d = {}
    for i in tags:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    
    tags = list(set(tags))
    return [tags, d]

# Computes the domain embedding through a weighted average of domain tags and their frequencies
def get_domain_embedding(glove_embedding, tags):
    embeddings = []
    counts = []
    for word, count in tags.items():
        if word in glove_embedding:
            embeddings.append(glove_embedding[word])
            counts.append(count)
        else:
            # If the word is not in the GloVe embedding, skip it
            continue
    
    #Take a weighted average of the embeddings using the word counts
    domain_embedding = sum(embeddings[i] * counts[i] for i in range(len(counts))) / sum(counts)
    return np.array(domain_embedding)

# Creates embeddings for items in the Amazon domain and returns a tupe with id along with embedding e.g (3, [0.92,...])
def amazon_with_tag_emb(df, glove_embedding):
    tups = []
    for index,row in df.iterrows():
        cat = row['category']
        pattern = "[0123456789" + re.escape(string.punctuation) + "]"
        tags_clean = list(map(lambda x: re.sub(pattern, '', x.lower()), cat))
        tags_split = list(map(lambda x: x.split(), tags_clean))
        tags_words = list(chain.from_iterable(tags_split))
        tags_words = list(tags_words)
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in tags_words if word.lower() not in stop_words])
        tags = list(text.split(' '))
        embeddings = []
        for word in tags:
            if word in glove_embedding:
                embeddings.append(glove_embedding[word])
        embeddings = np.array(embeddings)
        item_tags_embedding = np.mean(embeddings, axis=0)
        tup = (row['asin'], item_tags_embedding)
        tups.append(tup)
    return tups

# Creates embeddings for items in the Books domain and returns a tupe with id along with embedding e.g (3, [0.92,...])
def books_with_tag_emb(df, glove_embedding):
    tups = []
    for index,row in df.iterrows():
        cat = row['tag']
        cat = cat.split(',')
        pattern = "[0123456789" + re.escape(string.punctuation) + "]"
        tags_clean = list(map(lambda x: re.sub(pattern, '', x.lower()), cat))
        tags_split = list(map(lambda x: x.split(), tags_clean))
        tags_words = list(chain.from_iterable(tags_split))
        tags_words = list(tags_words)
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in tags_words if word.lower() not in stop_words])
        tags = list(text.split(' '))
        tags = list(set(tags))
        embeddings = []
        for word in tags:
            if word in glove_embedding:
                embeddings.append(glove_embedding[word])
        embeddings = np.array(embeddings)
        item_tags_embedding = np.mean(embeddings, axis=0)
        tup = (row['item_id'], item_tags_embedding)
        tups.append(tup)
    return tups

# Creates embeddings for items in the Movielens domain and returns a tupe with id along with embedding e.g (3, [0.92,...])
def movielens_with_tag_emb(df, glove_embedding):
    tups = []
    for index,row in df.iterrows():
        cat = row['tag']
        cat = cat.split(',')
        pattern = "[0123456789" + re.escape(string.punctuation) + "]"
        tags_clean = list(map(lambda x: re.sub(pattern, '', x.lower()), cat))
        tags_split = list(map(lambda x: x.split(), tags_clean))
        tags_words = list(chain.from_iterable(tags_split))
        tags_words = list(tags_words)
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in tags_words if word.lower() not in stop_words])
        tags = list(text.split(' '))
        tags = list(set(tags))
        embeddings = []
        for word in tags:
            if word in glove_embedding:
                embeddings.append(glove_embedding[word])
        embeddings = np.array(embeddings)
        item_tags_embedding = np.mean(embeddings, axis=0)
        tup = (row['movieId'], item_tags_embedding)
        tups.append(tup)     
    return tups

# Calculates Cosine similarity between two embeddings
def cos_sim(emb1, emb2):
    if isinstance(emb1, float) or len(list(emb1)) == 0 or emb1 is None:
        return 0
    if isinstance(emb2, float) or len(list(emb2)) == 0 or emb2 is None:
        return 0
    result = 1 - spatial.distance.cosine(emb1, emb2)
    return result

# Finds the average of a list
def average(lst):
    return sum(lst) / len(lst)

# Calculates Pairwise similarity between pairs of items in a cross-domain setting
def calc_cross_domain_sim(domain1, domain2):
    cross_domain_pairs = list(product(domain1, domain2))
    sims = []
    for i in cross_domain_pairs:
        emb1 = i[0][1]
        emb2 = i[1][1]
        sim = cos_sim(emb1, emb2)
        sims.append(sim)

    cross_domain_sim = average(sims)
    return cross_domain_sim