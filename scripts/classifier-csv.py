import csv
import nltk
from nltk import *
from random import shuffle
from nltk.corpus import stopwords
import pickle

def document_features(document):
	document_words = set(document)
   	features = {}
   	for word in word_features:
   	    features['contains(%s)' % word] = (word in document_words)
   	return features

tokenizer = RegexpTokenizer(r'\w+')
ps = PorterStemmer()

f = open('sentimentdataset.csv')
csv_f = csv.reader(f)

docs=[]
pos_docs=[]
neg_docs=[]
text = []
firstline = True
i=0
for row in csv_f:
	txt = unicode(row[3],errors='replace')
	if firstline:
		firstline = False
		continue
	if row[1]=='1':
		tokens = tokenizer.tokenize(txt) #split text in tokens
		filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
		stemed_tokens = [ps.stem(word) for word in filtered_tokens]
		pos_docs.append((list(filtered_tokens),'pos'))
	if row[1]=='0':
		tokens = tokenizer.tokenize(txt) #split text in tokens
		filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
		stemed_tokens = [ps.stem(word) for word in filtered_tokens]
		neg_docs.append((list(filtered_tokens),'neg'))
	text.extend(stemed_tokens)
	
	i=i+1
	if i==20000:
		break

print "Time to shuffle"
shuffle(pos_docs)
shuffle(neg_docs)

all_words = nltk.FreqDist(w.lower() for w in text )
word_features = all_words.keys()[:4000] + all_words.keys()[-4000:]

posfeaturesets = [(document_features(d), c) for (d, c) in pos_docs] #extract features that will be use for training and testing
negfeaturesets = [(document_features(d), c) for (d, c) in neg_docs] #extract features that will be use for training and testing

train_set = posfeaturesets[:4500] + negfeaturesets[:4500] #create training set from equal number of pos and neg docs
test_set = posfeaturesets[4500:] + negfeaturesets[4500:] #create testing set from equal number of pos and neg docs

classifier = nltk.NaiveBayesClassifier.train(train_set) #train the classifier using the training set
print nltk.classify.accuracy(classifier, test_set) #print the accuracy of our classifier using the testing set
classifier.show_most_informative_features(10) #show top 10 most informative features
save_classifier = open("naivebayes-csv.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()