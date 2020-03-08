from FeatureSets import part_of_speech_bigram as bipos
from FeatureSets import part_of_speech_unigram as unipos
from HelperFunctions import yelp_dataset_functions as yelp
from HelperFunctions import undersampling as us
from symspellpy.symspellpy import SymSpell, Verbosity
import numpy as np

X = np.zeros((5, 4))
print(X)
X[0] = X[0] + [1, 2, 3, 4]
print(X)