import json
import numpy as np
import sys
from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2

def vectorize_corpus_train(X, Y):
    vectorizer = CountVectorizer(stop_words='english', min_df=1,
                                 ngram_range=(1,2), max_features=2**16)
    X = vectorizer.fit_transform(X)

    tfidf = TfidfTransformer(use_idf=False)
    X = tfidf.fit_transform(X, np.transpose(np.array(Y)))

    return X.toarray()

def output_reviews(Y, X, outfile):
    star_mapping = {i + 1: ','.join('1' if i == idx else '0' for idx in range(5)) for i in range(5)}

    num_lines = len(X)
    elements = 0
    for review in X:
        elements += 1
        for val in review:
            if val != 0:
                elements += 1
    print(num_lines)
    print(elements)

    for idx, review in enumerate(X):
        review_data = []
        for i, val in enumerate(review):
            if val == 0:
                continue
            review_data.append(':'.join([str(i + 1), str(val)]))
        out_str = str(Y[idx]) + ' ' + ' '.join(review_data)
        print(out_str)

def main(training_infile, training_outfile):
    X_train = []
    Y_train = []

    t0 = time()
    for obj in open(training_infile):
        review = json.loads(obj)
        Y_train.append(review['stars'])
        X_train.append(review['text'])
    duration = time() - t0
    # print("done with loading data in %fs" % duration)
    # print("n: %d" % len(X_train))

    t0 = time()
    X_train = vectorize_corpus_train(X_train, Y_train)
    X_train = X_train.astype('float')
    duration = time() - t0
    # print("done with featurizing in %fs" % duration)
    # print("n_samples: %d, n_features: %d" % X_train.shape)

    output_reviews(Y_train, X_train.tolist(), training_outfile)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: featurize <training-input> <training-output>")
        exit(1)
    main(*sys.argv[1:])
