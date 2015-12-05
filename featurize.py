import json
import numpy as np
import sys
from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2

def vectorize_corpus(X, Y, X_test):
    vectorizer = CountVectorizer(stop_words='english', min_df=1,
                                 ngram_range=(1,2), max_features=2**16)
    X = vectorizer.fit_transform(X)
    X_test = vectorizer.transform(X_test)

    tfidf = TfidfTransformer(use_idf=False)
    X = tfidf.fit_transform(X, np.transpose(np.array(Y)))
    X_test = tfidf.transform(X_test)
    """
    feature_selector = LinearSVC(penalty="l1", dual=False, tol=1e-3)
    X = feature_selector.fit_transform(X, np.transpose(np.array(Y)))
    X_test = feature_selector.transform(X_test)
    """
    """
    ch2 = SelectKBest(chi2, k=2**12)
    X = ch2.fit_transform(X, np.transpose(np.array(Y)))
    X_test = ch2.transform(X_test)
    """

    return (X.toarray(), X_test.toarray())

def output_reviews(Y, X, outfile, X_test, testing_outfile):
    star_mapping = {i + 1: ','.join('1' if i == idx else '0' for idx in range(5)) for i in range(5)}
    for x, out in [(X, open(outfile, 'w')), (X_test, open(testing_outfile, 'w'))]:
        for idx, review in enumerate(x):
            review_data = []
            for i, val in enumerate(review):
                if val == 0:
                    continue
                review_data.append(':'.join([str(i + 1), str(val)]))
            out_str = str(Y[idx]) + ' ' + ' '.join(review_data) + '\n'
            out.write(out_str)

def main(training_infile, training_outfile, testing_infile, testing_outfile):
    X_train = []
    Y_train = []
    X_test = []

    t0 = time()
    for obj in open(training_infile):
        review = json.loads(obj)
        Y_train.append(review['stars'])
        X_train.append(review['text'])
    for obj in open(testing_infile):
        review = json.loads(obj)
        X_test.append(review['text'])
    duration = time() - t0
    print("done with loading data in %fs" % duration)
    print("n: %d" % len(X_train))

    t0 = time()
    X_train, X_test = vectorize_corpus(X_train, Y_train, X_test)
    X_train = X_train.astype('float')
    X_test = X_test.astype('float')
    duration = time() - t0
    print("done with featurizing in %fs" % duration)
    print("n_samples: %d, n_features: %d" % X_train.shape)

    output_reviews(Y_train, X_train.tolist(), training_outfile, X_test, testing_outfile)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: featurize <training-input> <training-output> <testing-input> <testing-output>")
        exit(0)
    main(*sys.argv[1:])

