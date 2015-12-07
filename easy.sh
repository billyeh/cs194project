./libsvm-3.20/svm-predict test.out <(python featurize.py data/review_short_train.json train.out data/review_short_test.json test.out | ./libsvm-3.20/svm-train -c 32 -g 0.03125 blah) predict.out

exit

echo "*** Featurizing ***"
time python featurize.py data/review_short_train.json train data/review_short_test.json test

if [[ $* == *--cross-validate* ]]
then
  echo "*** Cross validation ***"
  cd libsvm-3.20/tools
  time python grid.py ../../train ../../test | tail -n1 > ../../params
  cd ../..
  C=$(cat params | cut -d' ' -f1)
  g=$(cat params | cut -d' ' -f2)
else
  echo "*** Skipping cross validation ***"
  C="32.0"
  g="0.03125"
fi

echo "C=$C"
echo "g=$g"

echo "*** Training ***"
time ./libsvm-3.20/svm-train -c $C -g $g train
echo "*** Predicting ***"
time ./libsvm-3.20/svm-predict test train.model out
