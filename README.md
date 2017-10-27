# Word Disambiguation

## Problem
Our problem is to correct or disambugate the definiteness of a phrase. The problem
can be thought of as a binary classification problem. Our disambiguated text is A Tale of Two Cities from Charles Dickens. We trained on the text Oliver Twist which is also
from Dickens expecting there to be some stylistic similarities

## Algorithm
The structure of an NLP classification system is generally as follows:
• Extract a set of linguistic features relevant to predicting the output class
• Retrieve the corresponding vector for each feature
• Combine the vectors into an input vector x
• Feed the input vector into a non-linear classifier
We will discuss the first three points when we mention Data Preperation. For the classifer we chose a
Neural Network based classifier. Deep Neural Networks, or simply neural networks with more than two
hidden layers, are currently one of the most actively researched ML algorithms. They have shown
tremendous empirical performance on classification tasks involving Image Recognition and Natural
Language Processing . Two families of deep nets relevant to the problem are Convolutional Neural
Networks (Convnets) and Recurrent Neural Networks (RNNs). Most of the literature on text
classification uses RNNs with gated nerual units (LSTMs or GRUs) to solve the problem.
We solve the problem using RNNs built with LSTMs. LSTMs are simply gated neurons that deal with
the vanishing gradient problem inherent in deep networks by mantaining a steady flow of gradient.
The model we use is composed of a single LSTM layer followed by a mean pooling layer and a logistic
regression classifier on top. From the input sequence x 0 ,x 1 , x 2 .... , x k the LSTM layer will produce a
representation h 0 , h 1 , h 2 , ..., h k which is averaged in the pooling layer to h. This is then fed to a logistic
regression classifier.

## Data Preparation
Data Preparation and features play a prominent role in any ML task. In NLP tasks, a feature is
used to refer to a linguistic input such as a word, a suffix or a part-of-speech (POS) tag. For this task
we have used words as features used to predict the class. In order to do that we take our training corpus
and using a Tokenizer , the NLTK Punct Tokenizer, we break down the text into sentences. We only
choose sentences which have ‘a’ or ‘the’. If a sentence has multiple occurrances we break it down intophrases with one occurance per phrase e.g. The sentence “ When he was left alone, this strange being
took up a candle, went to a glass that hung against the wall, and surveyed himself minutely in it..” is
broken down to “When he was left alone, this strange being took up a candle, went to,” “a glass that
hung against” , “the wall and surveyed himself minutely in it”. It was a not straightforward how to best
break up the sentence. We considered POS tagging and using that but in the interest of time did not
pursue the approach.
Once we have the phrases we remove the article form from it and construct the label vector. We
tokenize the phrase into words and remove non-alphanumeric characters. Once we have our feature
vector we construct a dictionary of all the words in our features. We encode the features into indices
and the then associate an embedding corresponding to each index. This embedding matrix is our input
to the system.

## Implementation and Running
We programmed the problem in Python 2.7 using Theano to program the RNN. We also used
the NLTK package on Python. To run the code from first install Theano and NLTK and then from the
NLPTask folder you should run the ‘runRnn.py “LABELFILENAME”.txt
“OBFUSCATEDFILENAME”.txt ’. These files should be in the data folder. The program works if no
argument is given. You can run it on a different file. The ‘runRnn.py file uses parameters from a model
we trained. To train your own model you need to trainRnn.py and modify the ‘saveto’ parameter. Then
change the parameter_file parameter in ‘testLSTM’ fuction in ‘runRnn.py’

## Performance
As mentioned we used the novel ‘Oliver Twist’ to train. We divided the set into 75% training ,
5% validation and 75% training for a total 18671 train examples, 1245 valid examples and 4979 test
examples. We trained without a GPU and the average training time was about 3 hours due to an early
stopping mechanism. The best performance for the model error was 0.26 % training , 7.91 % test and
7.79 % validation error. However this did not yield the best performance on the given corpus. The
lowest error we achieved on the corpus was 26.41 % using a model with 22% training error. This the
model that we have provided with the code.

## Improvements
We identified a few avenues of improvement for the accuracy of the model
• Using a POS tag to both get phrases and as features alongwith words
• Using word2Vec or Glove embedding on the corpus to utilize similarity information. I started
with glove but the training was taking too long as I did not have access to GPUs.
• Experimenting with model parameters specially gradient schemes, learning rates and model size•
Include Precision, recall values and the confusion matrix

## Deliverables
We have included the code, the model and all files needed to run other than the libraries which
need to be installed. We have attached the ‘Output.txt’ and ‘Output.tsv’ file as required. This challenge
took us a full day of work.

## References
Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8),
1735-1780
Graves, Alex. Supervised sequence labelling with recurrent neural networks. Vol. 385.
Springer, 2012.
Bastien, Frédéric, Lamblin, Pascal, Pascanu, Razvan, Bergstra, James, Goodfellow, Ian,
Bergeron, Arnaud, Bouchard, Nicolas, and Bengio, Yoshua. Theano: new features and speed
improvements. NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2012.
Bergstra, James, Breuleux, Olivier, Bastien, Frédéric, Lamblin, Pascal, Pascanu, Razvan,
Desjardins, Guillaume, Turian, Joseph, Warde-Farley, David, and Bengio, Yoshua. Theano: a
CPU and GPU math expression compiler. In Proceedings of the Python for Scientific Computing
Conference (SciPy), June 2010.
