import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

import sys

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk

class Document(NamedTuple):
	doc_id: int
	news_category: List[str]
	data: List[str]

	def sections(self):
		return [self.news_category, self.data]

	def __repr__(self):
		return (f"doc_id: {self.doc_id}\n" +
			f"  news_category: {self.news_category}\n" +
			f"  data: {self.data}\n")

stemmer = SnowballStemmer('english')

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(file) as f:
        i = 0
        dev_start = 0
        for line in f:
            line = line.strip()

            token_line = word_tokenize(line)

            
            #Extracting the line from the JSON file, we want to extract the category as the label
            #and the rest as the data for that category
            docs.append(defaultdict(list))
            token_identifier = 0
            if token_line[6] == 'POLITICS':
            	token_identifier = 1
            else:
            	token_identifier = 0
            docs[i]['L'] = token_identifier
            for j in range(len(token_line)):
                token_line[j] = token_line[j].lower()
            idx_1 = token_line.index("headline")
            idx_2 = token_line.index("authors")
            idx_1 = idx_1 + 4
            docs[i]['T'] = token_line[idx_1:idx_2-3]
            #print(docs[i]['T'])
            i += 1

    return [Document(i + 1, d['L'], d['T'])
        for i, d in enumerate(docs[1:])]

def stem_doc(doc: Document):
    return Document(doc.doc_id, doc.news_category , *[[stemmer.stem(word) for word in doc.data]])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def compute_tf_uniform(docs: List[Document]):
    '''
    Computes smooth weighted term frequency total
    '''
    freq1 = Counter()
    freq2 = Counter()
    #freq3 = Counter()
    #freq4 = Counter()
    #freq5 = Counter()

    for doc in docs:
        if doc.news_category == 1:
            for term in doc.data:
                    freq1[term] += 1
        '''
        if doc.label == 'ENTERTAINMENT':
            for term in doc.txt:
                    freq2[term] += 1
        if doc.label == 'FOOD & DRINK':
            for term in doc.txt:
                    freq3[term] += 1
        if doc.label == 'SPORTS':
            for term in doc.txt:
                    freq4[term] += 1
        '''
        #Other category other than these 4.
       	if doc.news_category == 0:
       		for term in doc.data:
       				freq2[term] += 1
    return freq1, freq2

def compute_single_tf_uniform(doc: Document):
    '''
	Computes smooth weighted term frequency for one document
	'''
    freq = Counter()
    for term in doc.data:
        freq[term] += 1
    return freq

def get_centroid(total_of_label, doc_freqs):
	centroid = Counter()
	for term,weight in doc_freqs.items():
		centroid[term] = weight / total_of_label
	return centroid

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))

def experiment():
    file = sys.argv[1]

    docs = read_docs(file + '.-train.tsv')
    queries = read_docs(file + '.-dev.tsv')

    processed_docs, processed_queries = process_docs_and_queries(docs, queries, True, False)

    category1_total = 0
    category2_total = 0
    #category3_total = 0
    #category4_total = 0
    #category5_total = 0

    for doc in processed_docs:
        if doc.news_category == 1:
            category1_total += 1
        '''
        if doc.label == 'ENTERTAINMENT':
            category2_total += 1
        if doc.label == 'FOOD & DRINK':
            category3_total += 1
        if doc.label == 'SPORTS':
            category4_total += 1
        '''
        if doc.news_category == 0:
        	#For other categories
        	category2_total += 1

    category1_docs, category2_docs = compute_tf_uniform(docs)
    v_profile1 = get_centroid(category1_total, category1_docs)
    v_profile2 = get_centroid(category2_total, category2_docs)
    #v_profile3 = get_centroid(category3_total, category3_docs)
 	#v_profile4 = get_centroid(category4_total, category4_docs)
  	#v_profile5 = get_centroid(category5_total, category5_docs)


    total_correct = 0
    total = 0
    for query in processed_queries:
        #query_vec = compute_single_tf_smooth(query)
        #query_vec = compute_single_tf_step(query)
        #query_vec = compute_single_tf_mine(query)
        query_vec = compute_single_tf_uniform(query)
        sim1 = cosine_sim(query_vec, v_profile1)
        sim2 = cosine_sim(query_vec, v_profile2)
        #sim3 = cosine_sim(query_vec, v_profile3)
        #sim4 = cosine_sim(query_vec, v_profile4)
        #sim5 = cosine_sim(query_vec, v_profile5)

        if sim1 >= sim2 and query.news_category == 1:
            total_correct+=1
        if sim1 < sim2 and query.news_category == 0:
            total_correct+=1
        total+=1

    percent_correct = total_correct / total
    print("Percent correct: ", percent_correct)

def process_docs_and_queries(docs, queries, stem, adj):
    processed_docs = docs
    processed_queries = queries
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    if adj:
        processed_docs = spec_adj_tokens(processed_docs)
        processed_queries = spec_adj_tokens(processed_queries)
    return processed_docs, processed_queries

if __name__ == '__main__':
	experiment()

