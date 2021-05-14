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

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class classifierClass():
    def train(self, trainX, trainY):
        trainingMX = np.zeros((len(trainX)-1, 3))
        for idx in range (len(trainX)-1):
            trainingMX[idx,0] = trainX[idx][0]
            trainingMX[idx,1] = trainX[idx][1]
            trainingMX[idx,2] = trainX[idx][2]
        #trainingMX = scale(trainXMatrix)
        
        ## SVM Model: 54% with 50,000/180,000 lines from data set
        #self.clf = svm.SVC(gamma='scale')
        
        ## K nearest neighbors 41% with 50,000/180,000 lines from data set
        #self.clf = KNeighborsClassifier();

        ## RBF SVM 54% with 50,000/180,000 lines from data set
        #self.clf = SVC(gamma=2, C=1)
        
        ## Decision Tree 54% with 50,000/180,000 lines from data set
        #self.clf = DecisionTreeClassifier(max_depth=5)
        
        ## Random Forest 54% with 50,000/180,000 lines from data set
        self.clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=3)

        self.clf.fit(trainingMX, trainY)


    def classify(self, testX):
        trainingMX = np.zeros((len(testX), 3))
        for idx in range (len(testX)):
            trainingMX[idx,0] = testX[idx][0]
            trainingMX[idx,1] = testX[idx][1]
            trainingMX[idx,2] = testX[idx][2]

        return self.clf.predict(trainingMX)


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
			token_identifier = token_line[6]
			if token_identifier != 'POLITICS' and token_identifier != 'ENTERTAINMENT' and token_identifier != 'SPORTS' and token_identifier != 'FOOD & DRINK':
				token_identifier = 'OTHER'
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
	freq3 = Counter()
	freq4 = Counter()
	freq5 = Counter()

	for doc in docs:
		if doc.news_category == 'POLITICS':
			for term in doc.data:
					freq1[term] += 1
		if doc.news_category == 'ENTERTAINMENT':
			for term in doc.data:
					freq2[term] += 1
		if doc.news_category == 'FOOD & DRINK':
			for term in doc.data:
					freq3[term] += 1
		if doc.news_category == 'SPORTS':
			for term in doc.data:
					freq4[term] += 1
		#Other category other than these 4.
		if doc.news_category == 'OTHER':
			for term in doc.data:
					freq5[term] += 1
	return freq1, freq2, freq3, freq4, freq5

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

def sentimentAnalysis(docs: List[Document], positive, negative):
	results = []
	for doc in docs:
		#First getting the sentiment value, as it appears in the docs and in the positive and negative text files
		sent_val = 0
		for idx in doc.data:
			if idx in positive:
				sent_val += 1
			if idx in negative:
				sent_val -= 1
		
		#Second, getting the longest sequences of words in the doc.data with some sort of sentiment attached to them
		sent_val_2 = 0
		curr_val = 0
		max_val = 0
		for idx in range(len(doc.data)):
			if doc.data[idx] in negative:
				curr_val += 1
				if doc.data[idx] in positive:
					if max_val < curr_val:
						max_val = curr_val
					curr_val = 0
			if doc.data[idx] in positive:
				curr_val += 1
				if doc.data[idx] in negative:
					if max_val < curr_val:
						max_val = curr_val
					curr_val = 0
		sent_val_2 = max_val

		#Finally, getting the # of times there is a switch in polarity between positive and negative tones
		sent_val_3 = 0
		for idx in range(len(doc.data)):
			if doc.data[idx] in positive:
				next = idx + 1
				if idx == len(doc.data)-1:
					sent_val += 0
				else:
					if doc.data[next] in negative:
						sent_val_3 += 1
			if doc.data[idx] in negative:
				next = idx + 1
				if idx == len(doc.data)-1:
					sent_val += 0
				else:
					if doc.data[next] in negative:
						sent_val_3 += 1
		temp_result = [sent_val, sent_val_2, sent_val_3]
		results.append(temp_result)
	return results

def categorizer(docs: List[Document]):
	results = []
	for doc in docs:
		if doc.news_category == 'POLITICS':
			results.append('POLITICS')
		if doc.news_category == 'ENTERTAINMENT':
			results.append('ENTERTAINMENT')
		if doc.news_category == 'FOOD & DRINK':
			results.append('FOOD & DRINK')
		if doc.news_category == 'SPORTS':
			results.append('SPORTS')
		if doc.news_category == 'OTHER':
			results.append('OTHER')

	return results

def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

def experiment():
	file = sys.argv[1]

	docs = read_docs(file + '.-train.tsv')
	queries = read_docs(file + '.-dev.tsv')
	processed_docs, processed_queries = process_docs_and_queries(docs, queries, True, False)
	#SENTIMENT ANALYSIS AND PROCESSING TO BE ABLE TO FIT, TRAIN, AND PREDICT CATEGORIES FOR NEW HEADLINES
	#ADD/DELETE 3 APOSTROPHES (BELOW)
	
	neg_words = read_stopwords("negative-words.txt")
	pos_words = read_stopwords("positive-words.txt")

	sent_val = sentimentAnalysis(docs, pos_words, neg_words)
	train_vals_class = categorizer(docs)

	sent_test = sentimentAnalysis(queries, pos_words, neg_words)
	test_vals_class = categorizer(queries)

	classifier = classifierClass()
	classifier.train(sent_val, train_vals_class)
	results = classifier.classify(sent_test)

	amt_corr = 0
	idx = 0

	for result in results:
		if idx != len(test_vals_class):
			if result == test_vals_class[idx]:
				amt_corr += 1
			idx += 1
	print(amt_corr/idx)

	#ADD/DELETE 3 APOSTROPHES (ABOVE)
	#FOR REGULAR "BAG OF WORDS" COSINE SIMILARITY VALUES
	
	#ADD/DELETE THE APOSTROPHES TO USE THESE
	'''
	category1_total = 0
	category2_total = 0
	category3_total = 0
	category4_total = 0
	category5_total = 0

	for doc in processed_docs:
		if doc.news_category == 'POLITICS':
			category1_total += 1
		if doc.news_category == 'ENTERTAINMENT':
			category2_total += 1
		if doc.news_category == 'FOOD & DRINK':
			category3_total += 1
		if doc.news_category == 'SPORTS':
			category4_total += 1
		if doc.news_category == 'OTHER':
			#For other categories
			category5_total += 1

	category1_docs, category2_docs, category3_docs, category4_docs, category5_docs = compute_tf_uniform(docs)
	
	v_profile1 = get_centroid(category1_total, category1_docs)
	v_profile2 = get_centroid(category2_total, category2_docs)
	v_profile3 = get_centroid(category3_total, category3_docs)
	v_profile4 = get_centroid(category4_total, category4_docs)
	v_profile5 = get_centroid(category5_total, category5_docs)

	total_correct = 0
	total = 0
	for query in processed_queries:
		sim_vec = []
		#query_vec = compute_single_tf_smooth(query)
		#query_vec = compute_single_tf_step(query)
		#query_vec = compute_single_tf_mine(query)
		query_vec = compute_single_tf_uniform(query)
		sim1 = cosine_sim(query_vec, v_profile1)
		sim_vec.append(sim1)
		sim2 = cosine_sim(query_vec, v_profile2)
		sim_vec.append(sim2)
		sim3 = cosine_sim(query_vec, v_profile3)
		sim_vec.append(sim3)
		sim4 = cosine_sim(query_vec, v_profile4)
		sim_vec.append(sim4)
		sim5 = cosine_sim(query_vec, v_profile5)
		sim_vec.append(sim5)

		#Calculating max cosine similarity value
		max_cos_val = max(sim_vec)

		if sim1 == max_cos_val and query.news_category == 'POLITICS':
			total_correct+=1
		if sim2 == max_cos_val and query.news_category == 'ENTERTAINMENT':
			total_correct+=1
		if sim3 == max_cos_val and query.news_category == 'FOOD & DRINK':
			total_correct+=1
		if sim4 == max_cos_val and query.news_category == 'SPORTS':
			total_correct+=1
		if sim5 == max_cos_val and query.news_category == 'OTHER':
			total_correct+=1
		total+=1

	percent_correct = total_correct / total
	print("Percent correct: ", percent_correct)
	'''
	#ADD/DELETE THE APOSTROPHES TO USE THESE

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

