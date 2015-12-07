import json
import numpy as np
import sys
from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2


def vectorize_corpus_train(X_test):
    vectorizer = CountVectorizer(stop_words='english', min_df=1,
                                 ngram_range=(1,2), max_features=2**16)
    X_test = vectorizer.transform(X_test)
    tfidf = TfidfTransformer(use_idf=False)
    X_test = tfidf.transform(X_test)
    return X_test


def output_reviews(X_test, testing_outfile):
    star_mapping = {i + 1: ','.join('1' if i == idx else '0' for idx in range(5)) for i in range(5)}

    x = X_test
    # '''
    # with open(testing_outfile, 'w') as out:
    #     for idx, review in enumerate(x):
    #         review_data = []
    #         for i, val in enumerate(review):
    #             if val == 0:
    #                 continue
    #             review_data.append(':'.join([str(i + 1), str(val)]))
    #         out_str = str(Y[idx]) + ' ' + ' '.join(review_data) + '\n'
    #         out.write(out_str)
    # '''

    for x in (X_test,):
        num_lines = len(x)
        elements = 0
        for review in x:
            elements += 1
            for val in review:
                if val != 0:
                    elements += 1
        print(num_lines)
        print(elements)

        for idx, review in enumerate(x):
            review_data = []
            for i, val in enumerate(review):
                if val == 0:
                    continue
                review_data.append(':'.join([str(i + 1), str(val)]))
            out_str = str(Y[idx]) + ' ' + ' '.join(review_data)
            print(out_str)

def main(testing_infile, testing_outfile):

    X_test = []

    t0 = time()

    for obj in open(testing_infile):
        review = json.loads(obj)
        X_test.append(review['text'])
    duration = time() - t0

    t0 = time()
    X_test = vectorize_corpus_train(X_test)
    X_test = X_test.astype('float')
    duration = time() - t0

    output_reviews(X_test, testing_outfile)



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: featurize <testing_infile> <testing_outfile>")
        exit(0)
    main(*sys.argv[1:])


