import nltk
from nltk import *
from nltk.corpus import movie_reviews,stopwords
from nltk.corpus import wordnet
from random import shuffle
import json
import pickle
import os
import re
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
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
        conf = choice_votes / len(votes)
        return conf

def document_features(document):
	document_words = set(document)
   	features = {}
   	for word in word_features:
   		features[word] = (word in document_words)
   	return features

def docs_preprocessing(docname,sentiment,text):
	docs = []
	for line in open(docname, 'r'):
		line = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', line, flags=re.MULTILINE)
		json_data = json.loads(line)['text'] #reading text from tweet
		tokens = tokenizer.tokenize(json_data) #split text in tokens
		filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]#removing stopwords
		stemed_tokens = [ps.stem(word) for word in filtered_tokens]#stemming
		lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]#lemmatization
		lemmatized_tokens = nltk.pos_tag(lemmatized_tokens) #return lemmatized_tokens
		for w in lemmatized_tokens:
			if w[1][0] in allowed_word_types:
				text.append(w[0].lower())#storing all the words for later feature extraction
		docs.append((list(filtered_tokens),sentiment)) #return filtered_tokens
	return docs

def docs_preprocessing2(docname,sentiment,text): #for positive/negative.txt
	docs = []
	for line in open(docname, 'r'):
		line = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', line, flags=re.MULTILINE)
		tokens = tokenizer.tokenize(line.decode('latin-1')) #split text in tokens
		filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]#removing stopwords
		stemed_tokens = [ps.stem(word) for word in filtered_tokens]#stemming
		lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]#lemmatization
		lemmatized_tokens = nltk.pos_tag(lemmatized_tokens)
		for w in lemmatized_tokens:
			if w[1][0] in allowed_word_types:
				text.append(w[0].lower())#storing all the words for later feature extraction
		docs.append((list(filtered_tokens),sentiment))
	return docs

CURRENT_DIR = os.path.dirname(__file__)
file_path = os.path.join(CURRENT_DIR, "features_set.txt")
f = open(file_path,"w")

tokenizer = RegexpTokenizer(r'\w+')
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
pos_docs=[]
neg_docs=[]
text = []
allowed_word_types = ["J","R","V"]

# pos_docs = docs_preprocessing("positive_tweets.json","pos",text)
# neg_docs = docs_preprocessing("negative_tweets.json","neg",text)

pos_docs = docs_preprocessing2("positive.txt","pos",text)
neg_docs = docs_preprocessing2("negative.txt","neg",text)

# for line in open('positive_tweets.json', 'r'):
#     json_data = json.loads(line)['text'] #reading text from tweet
#     tokens = tokenizer.tokenize(json_data.encode("utf-8")) #split text in tokens
#     filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]#removing stopwords
#     stemed_tokens = [ps.stem(word) for word in filtered_tokens]#stemming
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]#lemmatization
#     text.extend(lemmatized_tokens)#storing all the words for later feature extraction
#     pos_docs.append((list(filtered_tokens),'pos'))

# for line in open('negative_tweets.json', 'r'):
#     json_data = json.loads(line)['text'] #reading text from tweet
#     tokens = tokenizer.tokenize(json_data) #split text in tokens
#     filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]#removing stopwords
#     stemed_tokens = [ps.stem(word) for word in filtered_tokens]#stemming
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]#lemmatization
#     text.extend(lemmatized_tokens)#storing all the words for later feature extraction
#     neg_docs.append((list(filtered_tokens),'neg'))

all_words = nltk.FreqDist(w.lower() for w in text) #find the frequencies for all the unique words of our data
word_features = all_words.keys()[:4000] + all_words.keys()[-1000:] #from the previous list keep the 4000 most frequent and 4000 less frequent words

posfeaturesets = [(document_features(d), c) for (d, c) in pos_docs] #extract features that will be use for training and testing
negfeaturesets = [(document_features(d), c) for (d, c) in neg_docs] #extract features that will be use for training and testing

train_set = posfeaturesets[:4800] + negfeaturesets[:4800] #create training set from equal number of pos and neg docs
test_set = posfeaturesets[4800:] + negfeaturesets[4800:] #create testing set from equal number of pos and neg docs

shuffle(train_set)
shuffle(test_set)

# for word in word_features:
# 	f.write(word.encode('utf-8')+'\n')
# f.close

classifier = nltk.NaiveBayesClassifier.train(train_set) #train the classifier using the training set
accuracy = nltk.classify.accuracy(classifier, test_set) #print the accuracy of our classifier using the testing set
print "The accuracy for this round is: " , accuracy
classifier.show_most_informative_features(10) #show top 10 most informative features
if(accuracy > 0.7):
	print "Saving classifier..."
	save_classifier = open("naivebayes.pickle","wb")
	pickle.dump(classifier,save_classifier)

# documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
# # shuffle(documents)
# all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words() )
# word_features = all_words.keys()[:2000] + all_words.keys()[-2000:]
# for word in word_features:
# 	f.write(word+'\n')
# f.close
# featuresets = [(document_features(d), c) for (d, c) in documents]
# train_set, test_set = featuresets[:1900], featuresets[1900:] #divide the whole set in training and testing set
# classifier = nltk.NaiveBayesClassifier.train(train_set) #train the classifier using the training set
# accuracy = nltk.classify.accuracy(classifier, test_set) #print the accuracy of our classifier using the testing set
# print "The accuracy for this round is: " , accuracy
# classifier.show_most_informative_features(10) #show top 10 most informative features

# if(accuracy > 0.7):
# 	print "Saving classifier..."
# 	save_classifier = open("naivebayes.pickle","wb")
# 	pickle.dump(classifier,save_classifier)
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, test_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, test_set))*100)
print "Saving classifier..."
save_classifier = open("BernouliNB.pickle","wb")
pickle.dump(BernoulliNB_classifier,save_classifier)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100)
print "Saving classifier..."
save_classifier = open("LogisticReg.pickle","wb")
pickle.dump(LogisticRegression_classifier,save_classifier)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(train_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, test_set))*100)
print "Saving classifier..."
save_classifier = open("SGDC.pickle","wb")
pickle.dump(SGDClassifier_classifier,save_classifier)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)
print "Saving classifier..."
save_classifier = open("LinearSVC.pickle","wb")
pickle.dump(LinearSVC_classifier,save_classifier)

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(train_set)
# print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, test_set))*100)
# print "Saving classifier..."
# save_classifier = open("NuSVC.pickle","wb")
# pickle.dump(NuSVC_classifier,save_classifier)
# save_classifier.close()


voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, test_set))*100)

save_classifier = open("VoteClassifier.pickle","wb")
pickle.dump(voted_classifier,save_classifier)
save_classifier.close()


