import sys
import locale
import os
import nltk
import re
import collections
import string
import numpy
import theano
import six.moves.cPickle as pickle

DICTIONARY = 'Oliver_Twist'
# Dictionary of labels
labelMap = {
	'a' : 0,
	'the' : 1
	}

trainingFile = 'oliver_twist.txt'
# trainingFile = 'A_tale_of_two_cities.100kwords.obfuscated.txt'

def findWholeWord(word):
    return re.compile(r'\b({0})\b'.format(word), flags=re.IGNORECASE).finditer

def deleteArticle(article):
	return re.compile(r'\b({0})\b'.format(article)).sub

def lengthSort(seq):
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))

def saveObject(obj, name ):
    with open('Data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadObject(name ):
    with open('Data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def prepareData(seqs):
    """Create the matrices from the datasets.

    This pads each sequence to the same lenght: the lenght of the
    longuest sequence

    This swaps the axis!
    """
    lengths = [len(s) for s in seqs]

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1

    return x, x_mask

def getTrainingSet():
	
	features = []
	labels = []
	count = 0

	fp = open('./Data/' + trainingFile)
	data = fp.read().decode('utf-8').encode('ascii','ignore')
	
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = tokenizer.tokenize(data)

	for sentence in sentences:
		indices = []
		l = []
		for article in labelMap.keys():
			iterable = [i.start() for i in findWholeWord(article)(sentence)]
			indices.extend(iterable)
			
			if len(iterable) > 0:	
				l.extend(len(iterable) * [article])

		index_length = len(indices)
		previous = 0
		index_order = sorted(range(len(indices)), key=lambda k: indices[k])
		sorted_l = [l[i] for i in index_order]
		sorted_indices = [indices[i] for i in index_order]
		
		for index in range(0, index_length):
			
			if index < index_length -1:
				nxt = sorted_indices[index+1] 
			else:
				nxt = len(sentence)

			current = sorted_indices[index] + len(sorted_l[index]) 
			
			if index == 0:
				feature = sentence[:nxt]
			
			elif index == index_length -1:
				feature = sentence[previous:nxt]
			else:
				feature = sentence[current:nxt]

			feature = deleteArticle(sorted_l[index])('', feature.lower())
			pattern = re.compile('([^\s\w]|_)+')
			feature = pattern.sub(' ', feature)
			tokenizer = nltk.tokenize.WordPunctTokenizer()
			if len(tokenizer.tokenize(feature)) > 2:
				features.append(tokenizer.tokenize(feature))
				labels.append(labelMap[sorted_l[index]])
			previous = sorted_indices[index] + len(sorted_l[index])
			count = count + 1

	return features, labels

def buildDictionary(features):

	print 'Building dictionary..'

	wordcount = dict()

	for words in features:
		for word in words:
		    if word not in wordcount:
		        wordcount[word] = 1
		    else:
		        wordcount[word] += 1

	counts = wordcount.values()
	keys = wordcount.keys()

	sorted_idx = numpy.argsort(counts)[::-1]

	word_dict = dict()

	for idx, ss in enumerate(sorted_idx):
		word_dict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

	print numpy.sum(counts), ' total words ', len(keys), ' unique words'

	saveObject(word_dict, DICTIONARY)

	return word_dict

def encodeData(features, dictionary):

    seqs = [None] * len(features)
    for idx, words in enumerate(features):
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs

def loadData(n_words=50000, test_portion=0.2, valid_portion=0.05,
          sort_by_len=True):

	# Loads the dataset

	# type n_words: int
	# param n_words: The number of word to keep in the vocabulary.
	#     All extra words are set to unknown.

	# type test_portion: float
	# param valid_portion: The proportion of the full train set used for
	#     the validation set.

	# type test_portion: float
	# param valid_portion: The proportion of the full train set used for
	#     the validation set.

	# type sort_by_len: bool
	# name sort_by_len: Sort by the sequence lenght for the train,
	#     valid and test set. This allow faster execution as it cause
	#     less padding per minibatch. Another mechanism must be used to
	#     shuffle the train set at each epoch.


	features, data_y = getTrainingSet()
	dictionary = buildDictionary(features)
	data_x = encodeData(features, dictionary)

	
	# split data set into training, test and validation set

	n_samples = len(data_x)
	sidx = numpy.random.permutation(n_samples)
	n_test = int(numpy.round(n_samples * test_portion))
	n_valid = int(numpy.round(n_samples * valid_portion))
	n_non_training = n_test + n_valid 

	test_set_x = [data_x[s] for s in sidx[:n_test]]
	test_set_y = [data_y[s] for s in sidx[:n_test]]
	valid_set_x = [data_x[s] for s in sidx[n_test:n_non_training]]
	valid_set_y = [data_y[s] for s in sidx[n_test:n_non_training]]
	train_set_x = [data_x[s] for s in sidx[n_non_training:]]
	train_set_y = [data_y[s] for s in sidx[n_non_training:]]

	train_set = (train_set_x, train_set_y)
	test_set = (test_set_x, test_set_y)
	valid_set = (valid_set_x, valid_set_y)

	test_set_x, test_set_y = test_set
	valid_set_x, valid_set_y = valid_set
	train_set_x, train_set_y = train_set

	def removeUnknown(x):
		return [[1 if w >= n_words else w for w in sen] for sen in x]

	train_set_x = removeUnknown(train_set_x)
	valid_set_x = removeUnknown(valid_set_x)
	test_set_x = removeUnknown(test_set_x)


	if sort_by_len:
	    sorted_index = lengthSort(test_set_x)
	    test_set_x = [test_set_x[i] for i in sorted_index]
	    test_set_y = [test_set_y[i] for i in sorted_index]

	    sorted_index = lengthSort(valid_set_x)
	    valid_set_x = [valid_set_x[i] for i in sorted_index]
	    valid_set_y = [valid_set_y[i] for i in sorted_index]

	    sorted_index = lengthSort(train_set_x)
	    train_set_x = [train_set_x[i] for i in sorted_index]
	    train_set_y = [train_set_y[i] for i in sorted_index]

	train = (train_set_x, train_set_y)
	valid = (valid_set_x, valid_set_y)
	test = (test_set_x, test_set_y)
	return train, valid, test

if __name__ == '__main__':
	loadData()

