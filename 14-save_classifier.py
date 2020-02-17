import nltk 
import random
from nltk.corpus import movie_reviews
import pickle

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

#print(documents[1])
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Naive-bayes -  posterior = prior occurances x(times) liklihood / evidence - Scalable, small algo which doesn't require much computation.

#classifier = nltk.NaiveBayesClassifier.train(training_set)
#### Load the classifier
classifier_f = open("naivebayes-14.pickle", 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

#### Save the classifier
# save_classifier = open("naivebayes-14.pickle", 'wb')
# pickle.dump(classifier, save_classifier)