if [ -z "$1" ]
then
    echo "Usage: bash create_files.sh <yelp_review.json>"
    exit 1
fi

gshuf "$1" -o review.json
sed 1,1000d review.json > review_train.json
head -n 1000 review.json > review_test.json
rm review.json

head -n 1000 review_train.json > review_1000.json
head -n 2000 review_train.json > review_2000.json
head -n 4000 review_train.json > review_4000.json
head -n 8000 review_train.json > review_8000.json
head -n 16000 review_train.json > review_16000.json
