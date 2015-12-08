Rough outline of code: generate test and training data using `data/create_files.sh` -> 
featurize using `featurize.py` (training and test data) -> train using 
`svmlib` (training data) -> predict using `svmlib` (test data).

For a simple benchmark, run `bash easy.sh`. This file contains commands for all 3 parts, 
using the different file sizes created in the `data/` directory using `create_files.sh`.
