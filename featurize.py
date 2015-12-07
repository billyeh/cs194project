import contextlib, itertools, json, os, shutil, subprocess, sys, tempfile

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


@contextlib.contextmanager
def named_pipe():
    dirname = tempfile.mkdtemp()
    try:
        path = os.path.join(dirname, 'fifo')
        os.mkfifo(path)
        yield path
    finally:
        shutil.rmtree(dirname)


def vectorize_corpus(X_train, Y_train, X_test):
    vectorizer = CountVectorizer(stop_words='english', min_df=1,
                                 ngram_range=(1,2), max_features=2**16)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    tfidf = TfidfTransformer(use_idf=False)
    X_train = tfidf.fit_transform(X_train, np.transpose(np.array(Y_train)))
    X_test = tfidf.transform(X_test)

    return X_train.toarray(), X_test.toarray()


def output_reviews(X, Y, prints_num_lines=False, prints_num_elems=False,
                   stdout=sys.stdout):
    if prints_num_lines:
        stdout.write(str(len(X)) + '\n')
    if prints_num_elems:
        stdout.write(str(len(X) + np.count_nonzero(X)) + '\n')

    for label, review in itertools.izip(Y, X):
        line = str(label)
        for j in review.nonzero()[0]:
            line += ' {0}:{1}'.format(j + 1, review[j])
        stdout.write(line + '\n')


def main(training_infile, test_infile):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    with open(training_infile, 'r') as f:
        for line in f:
            review = json.loads(line)
            X_train.append(review['text'])
            Y_train.append(review['stars'])
    with open(test_infile, 'r') as f:
        for line in f:
            review = json.loads(line)
            X_test.append(review['text'])
            Y_test.append(review['stars'])

    X_train, X_test = vectorize_corpus(X_train, Y_train, X_test)

    proc_train = subprocess.Popen(['./libsvm-3.20/svm-train',
                                   '-c', '32', '-g', '0.03125',
                                   '/dev/null'],
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE)
    output_reviews(X_train, Y_train,
                   prints_num_lines=True, prints_num_elems=True,
                   stdout=proc_train.stdin)
    proc_train.stdin.close()

    with named_pipe() as test_path, named_pipe() as model_path:
        proc_predict = subprocess.Popen(['./libsvm-3.20/svm-predict',
                                         test_path,  # test file
                                         model_path, # model file
                                         'predict.out'])

        with open(model_path, 'w+') as model_pipe:
            model_pipe.write(proc_train.stdout.read())
        with open(test_path, 'w+') as test_pipe:
            output_reviews(X_test, Y_test,
                           prints_num_lines=True, prints_num_elems=False,
                           stdout=test_pipe)

    print('done')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: featurize <training-input> <test-input>")
        exit(1)
    main(*sys.argv[1:])
