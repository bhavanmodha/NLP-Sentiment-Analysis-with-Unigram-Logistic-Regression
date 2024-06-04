import re
import sys

import nltk
import numpy
from sklearn.linear_model import LogisticRegression


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
    corpus=[]
    file = open(corpus_path,'r')
    for lines in file:
        tokens = lines.strip().split("\t")
        corpus.append((tokens[0].split(" "),int(tokens[1])))
    return corpus


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    if word in negation_words or word[-3:]=="n't":
        return True
    return False
        


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    tagged_snippet = nltk.pos_tag(snippet)
    for i in range(len(tagged_snippet)):
        word = tagged_snippet[i][0]
        
        if is_negation(word) and word + tagged_snippet[i+1][0]!="notonly":
            for j in range(i+1,len(tagged_snippet)):
                temp = tagged_snippet[j][0]
                tag = tagged_snippet[j][1]
                if temp in negation_enders or temp in sentence_enders or tag in ["JJR", "RBR"]:
                    break
                snippet[j] = "NOT_"+snippet[j]
        break
    return snippet


# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    feature_dict = {}
    c = 0
    for snippet, class_ in corpus:
        for word in snippet:
            if word not in feature_dict:
                feature_dict[word] = c
                c+=1
    return feature_dict
    

# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    input_vector = numpy.zeros(len(feature_dict))
    for word in snippet:
        if word in feature_dict:
            input_vector[feature_dict[word]]+=1
    return input_vector


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    features = numpy.empty([len(corpus),len(feature_dict)])
    result = numpy.empty(len(corpus))
    for idx, (snippet, class_) in enumerate(corpus):
        features[idx] = vectorize_snippet(snippet, feature_dict)
        result[idx] = class_
    return (features,result)


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    col_max = X.max(axis=0)
    col_min = X.min(axis=0)
    for i in range(len(X)):
        for j in range(len(X[0])):
            num = X[i][j]-col_min[j]
            deno = col_max[j]-col_min[j]
            if deno>0:
                X[i][j] = num/deno
    return X


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    corpus = load_corpus(corpus_path)
    for i in range(len(corpus)):
        corpus[i] = (tag_negation(corpus[i][0]),corpus[i][1])
    feature_dict = get_feature_dictionary(corpus)
    vect_corp = vectorize_corpus(corpus, feature_dict)
    features = normalize(vect_corp[0])
    results = vect_corp[1]

    logreg_model = LogisticRegression()
    logreg_model.fit(features, results)

    return (logreg_model,feature_dict)


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    true_pos=0
    false_pos=0
    false_neg=0
    for i in range(len(Y_pred)):
        if Y_test[i]==1 and Y_pred[i]==1:
            true_pos+=1
        elif Y_test[i]==0 and Y_pred[i]==1:
            false_pos+=1
        elif Y_test[i]==1 and Y_pred[i]==0:
            false_neg+=1
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    f_measure = 2*(precision*recall)/(precision+recall)
    return (precision, recall, f_measure)


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    corpus = load_corpus(corpus_path)
    for i in range(len(corpus)):
        corpus[i] = (tag_negation(corpus[i][0]),corpus[i][1])
    vect_corp = vectorize_corpus(corpus, feature_dict)
    test_features = normalize(vect_corp[0])
    test_results = vect_corp[1]
    pred = model.predict(test_features)
    return evaluate_predictions(pred, test_results)


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    k_features = []
    w = logreg_model.coef_[0]
    idx_w=[]
    for i in range(len(w)):
        idx_w.append((i,w[i]))
    idx_w = sorted(idx_w, key=lambda x: abs(x[1]), reverse=True)
    words = [0]*len(feature_dict)
    for i in feature_dict:
        words[feature_dict[i]] = i
    for i in range(k):
        k_features.append((words[idx_w[i][0]],idx_w[i][1]))
    return k_features


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
