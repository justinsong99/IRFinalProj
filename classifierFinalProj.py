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
	title: List[str]
	news_category: List[str]

	def sections(self):
		return [self.title, self.news_category]

	def __repr__(self):
		return (f"doc_id: {self.doc_id}\n" +
			f"  title: {self.title}\n" +
			f"  news_category: {self.classlabel}\n")

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

            if len(docs) == 1 and token_line[0] != 1:
                dev_start = int(token_line[0]) - 1

            i = int(token_line[0]) - dev_start
            #Extracting the line from the JSON file, we want to extract the category as the label
            #and the rest as the data for that category
            docs.append(defaultdict(list))
            docs[i]['L'] = int(token_line[1])
            for j in range(len(token_line)):
                token_line[j] = token_line[j].lower()
            docs[i]['T'] = token_line[2:]

    return [Document(i + 1, d['L'], d['T'])
        for i, d in enumerate(docs[1])]

def stem_doc(doc: Document):
    return Document(doc.doc_id, doc.label , *[[stemmer.stem(word) for word in doc.txt]])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def compute_tf_uniform(docs: List[Document]):
    '''
    Computes smooth weighted term frequency total
    '''
    freq1 = Counter()
    freq2 = Counter()
    freq3 = Counter()
    freq4 = Counter()
    freq5 = Counter()

    for doc in docs:
        if doc.label == 'POLITICS':
            for term in doc.txt:
                    freq1[term] += 1
        if doc.label == 'ENTERTAINMENT':
            for term in doc.txt:
                    freq2[term] += 1
        if doc.label == 'FOOD & DRINK':
            for term in doc.txt:
                    freq3[term] += 1
        if doc.label == 'SPORTS':
            for term in doc.txt:
                    freq4[term] += 1
        #Other category other than these 4.
       	else:
       		for term in doc.txt:
       				freq5[term] += 1
    return freq1, freq2, freq3, freq4

def experiment():
    file = sys.argv[1]

    docs = read_docs(file[:-4] + '-train.tsv')
    queries = read_docs(file[:-4] + '-dev.tsv')

    processed_docs, processed_queries = process_docs_and_queries(docs, queries, True, False)

    category1_total = 0
    category2_total = 0
    category3_total = 0
    category4_total = 0
    category5_total = 0

    for doc in processed_docs:
        if doc.label == 'POLITICS':
            category1_total += 1
        if doc.label == 'ENTERTAINMENT':
            category2_total += 1
        if doc.label == 'FOOD & DRINK':
            category3_total += 1
        if doc.label == 'SPORTS':
            category4_total += 1
        else:
        	#For other categories
        	category5_total += 1

    category1_docs, category2_docs, category3_docs, category4_docs, category5_docs = compute_tf_uniform(docs)
    v_profile1 = get_centroid(label1_total, label1_docs)
    v_profile2 = get_centroid(label2_total, label2_docs)

    total_correct = 0
    total = 0
    for query in processed_queries:
        #query_vec = compute_single_tf_smooth(query)
        #query_vec = compute_single_tf_step(query)
        #query_vec = compute_single_tf_mine(query)
        query_vec = compute_single_tf_uniform(query)
        sim1 = cosine_sim(query_vec, v_profile1)
        sim2 = cosine_sim(query_vec, v_profile2)

        if sim1 >= sim2 and query.label == 1:
            total_correct+=1
        if sim1 < sim2 and query.label == 2:
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

