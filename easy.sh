echo "*** Featurizing ***"
python featurize.py data/review_short_train.json train data/review_short_test.json test

echo "*** Cross validation ***"
cd libsvm-3.20/tools
time python grid.py ../../train ../../test | tail -n1 | tee ../../params
cd ../..

echo "*** Training and predicting ***"
C=$(cat params | cut -d' ' -f1)
g=$(cat params | cut -d' ' -f2)

echo "C=$C"
echo "g=$g"

./libsvm-3.20/svm-train -c $C -g $g train
./libsvm-3.20/svm-predict test train.model out
