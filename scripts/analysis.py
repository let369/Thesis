import sys
import pickle
import os
from nltk.corpus import wordnet
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
        	v = c.classify(features)
        	votes.append(v)
        	choice_votes = votes.count(mode(votes))
        	conf = float(choice_votes) / len(votes)
        return conf

def extract_features(document): #creates features based on the extracted features from the training requires a file with the words
	document_words = set(document)
   	features = {}
   	for word in word_features:
   	    features[word] = (word in document_words)
   	# print features
   	return features

def bag_of_words(words): #creates features from all the words of our sentence
	features = {}
	for word in words:
		features[word] = True
		# syns = wordnet.synsets(word)
		# for w in syns:
		# 	features[w.lemmas()[0].name()] = True
	return features
    # return dict([word, True] for word in words)

CURRENT_DIR = os.path.dirname(__file__)

sentiment = ""
text = []
word_features = []
for i in range(1,len(sys.argv)):
	text.append(sys.argv[i])
# bag_of_words(text)
# file_path = os.path.join(CURRENT_DIR, 'features_set.txt')
# file = open(file_path,"r")
# for line in file:
# 	word =  line.rstrip('\n')
# 	word_features.append(word)
# print "Features were read from file."

file_path = os.path.join(CURRENT_DIR, 'originalnaivebayes5k.pickle')
classifier_f = open(file_path,"rb")
naivebayes_classifier = pickle.load(classifier_f)

sentiment = naivebayes_classifier.classify(bag_of_words(text))
print sentiment



# file_path = os.path.join(CURRENT_DIR, 'VoteClassifier.pickle')
# classifier_f = open(file_path,"rb")
# vote_classifier = pickle.load(classifier_f)
# classifier_f.close()

# sentiment = vote_classifier.classify(bag_of_words(text))
# print sentiment
