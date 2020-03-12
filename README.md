# How to run

## Download the datasets
Download and unzip the datasets found at: https://drive.google.com/drive/folders/1CM4d54k9vgYHhkwb5nu94Z37MVKN3Kfw?usp=sharing. Simply put the Reviews folder in the root directory of the repository.


## Running the SVM
In the main.py file in the folder called main, the function execute_SVM_process() can be found. The first argument of this function can either be 'regular', '45stars' or '12stars'. These arguments will result in the SVM being trained on a sample containing all star ratings, only 4-5 star ratings and 1-2 star ratings respectively. 

The second argument is to select which feature set to use. The options are 'unigram', for unigram features, 'bigram' for bigram features, 'unipos' for unigram part of speech features, 'bipos' for bigram part of speech features, 'ig1%' for top 1% information gain features, 'ig2%' for top 2% information gain features, 'deep' for deep syntax features (NOTE: you will need to have the Stanford CoreNLP server running on port 9500 in order to run deep syntax features) and lastly 'posseq' for part of speech sequence pattern features. 

If the third argument remains unchanged (False), no new sample will be generated and the SVM will be executed on the same balanced sample as the results found in the report. Setting this parameter to True will generate a new balanced sample.
