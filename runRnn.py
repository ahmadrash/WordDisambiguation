from __future__ import print_function
import six.moves.cPickle as pickle
from collections import OrderedDict
import nltk
import sys
import time
import numpy
import theano
import re
import string
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import definiteness
import trainRnn



# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)


def predProbs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data)
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)
    n_done = 0

    for _, valid_index in iterator:
        x, mask = prepare_data([data[t] for t in valid_index])
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs

def getTestSet(dataFile):

	features = []
	labels = []
	count = 0
	check = 0

	fp = open('./Data/' + dataFile)
	data = fp.read().decode('utf-8').encode('ascii','ignore')
	
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = tokenizer.tokenize(data)

	for sentence in sentences:
		indices = []
		l = []
		for article in definiteness.labelMap.keys():
			iterable = [i.start() for i in definiteness.findWholeWord(article)(sentence)]
			indices.extend(iterable)
			
			if len(iterable) > 0:	
				l.extend(len(iterable) * [article])

		index_length = len(indices)
		# previous = - 1
		index_order = sorted(range(len(indices)), key=lambda k: indices[k])
		sorted_l = [l[i] for i in index_order]
		sorted_indices = [indices[i] for i in index_order]
		
		for index in range(0, index_length):
			# print(indices)
			# print(index)
			# print(index_length)
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

			feature = definiteness.deleteArticle(sorted_l[index])('', feature.lower())
			pattern = re.compile('([^\s\w]|_)+')
			feature = pattern.sub(' ', feature)
			tokenizer = nltk.tokenize.WordPunctTokenizer()
			labels.append(definiteness.labelMap[sorted_l[index]])
			if len(tokenizer.tokenize(feature)) > 1:
				features.append(tokenizer.tokenize(feature))
			else:
				features.append(features[0])
				# check = check + 1
				# print(check)

			previous = sorted_indices[index] + len(sorted_l[index])
			count = count + 1

	# print(features[0:20])
	# print(labels[0:20])
	# print(len(features))
	# print(len(labels))
	return features, labels


def testLSTM(
	dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=10000,  # Vocabulary size
    optimizer=trainRnn.adadelta,  # adadelta or rmsprop
    encoder='lstm',
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1000,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set
    parameter_file = 'lstm_model_4.npz',
    labelFile = 'A_tale_of_two_cities.100kwords.original.txt',

    # Parameter for extra options
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
):
	model_options = locals().copy()

	features, labels = getTestSet(model_options['labelFile'])

	dictionary = definiteness.loadObject(definiteness.DICTIONARY)
	data_x = definiteness.encodeData(features, dictionary)

	def removeUnknown(x):
		return [[1 if w >= n_words else w for w in sen] for sen in x]

	data_x = removeUnknown(data_x)

	prepare_data = definiteness.prepareData
	build_model = trainRnn.buildModel

	ydim = numpy.max(labels) + 1
	model_options['ydim'] = ydim

	params = trainRnn.initParameters(model_options)
	params = trainRnn.loadParameters(model_options['parameter_file'], params)

	tparams = trainRnn.initTrainingParameters(params)

	(use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

	kf_test = trainRnn.getMinibatchesIdx(len(features), 1)
	probabilities = predProbs(f_pred_prob, prepare_data, data_x, kf_test)
	results = numpy.argmax(probabilities, axis = 1)
	
	wrong = 0.0 
	for x in range(0,len(results)):
		if results[x] == labels[x]:
			a = 0
		else:
			wrong = wrong + 1

	print('The accuracy is ' + str((1 - wrong/len(results) )* 100) + ' percent.')
	return results , labels

		# for key, value in definiteness.labelMap.iteritems():
		# 	if value == result:
		# 		print(key) 

def main():
	
	if len(sys.argv) == 3:
		label_file = sys.argv[1]
		test_file = sys.argv[2]

	else:
		label_file = 'A_tale_of_two_cities.100kwords.original.txt'
		test_File = 'A_tale_of_two_cities.100kwords.obfuscated.txt'

	results, labels = testLSTM(labelFile=label_file)
	count = 0
	
	with open('./Data/' + test_file, "r") as inFile, open("Output.txt", "w") as outFile, open("Output.tsv", "w") as tsvFile:
		
		for line in inFile:
			indices = []
    		
			for article in definiteness.labelMap.keys():
				iterable = [i.start() for i in definiteness.findWholeWord(article)(line)]
				indices.extend(iterable)
			
			sorted_indices = sorted(indices)

			if len(sorted_indices) == 0:
				for word in line.split():
					tsvFile.write("%s\t%s\t%s\n" % (word, word, word))

				outFile.write(line)

			else:
				new_line = ''
				previous = 0
				original_list = []
				change_list = []
				
				for index in sorted_indices:
	
					if line[index].lower() == 'a':
						length = 1
					else:
						length = 3

					new_line = new_line + line[previous:index]
					change = ''
					original = ''

					
					for article, value in definiteness.labelMap.items():
						if value == results[count]:
							change = article
						if value == labels[count]:
							original = article

					original_list.append(original)
					change_list.append(change) 
					new_line = new_line + change

					previous = index + length 
					count = count + 1

					if index == sorted_indices[-1]:
						new_line = new_line + line[previous:len(line)]

				outFile.write(new_line)

				for index, word in enumerate(line.split()):
					
					if word.lower() in ['a', 'the']:
						tsvFile.write("%s\t%s\t%s\n" % (original_list.pop(0), word, change_list.pop(0)))

					else:
						tsvFile.write("%s\t%s\t%s\n" % (word, word, word))

		outFile.close()
		inFile.close()
		tsvFile.close()

if __name__ == '__main__':
	main()
	# test()