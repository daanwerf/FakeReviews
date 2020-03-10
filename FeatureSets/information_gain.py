from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from nltk.stem import PorterStemmer
from symspellpy.symspellpy import SymSpell, Verbosity
from nltk.corpus import stopwords
from FeatureExtraction import bag_of_words
from scipy.stats import entropy
import numpy as np

def test_random_stuff():

    vectorizer_fake, speller_fake, stop_words_fake, ps_fake, top_words_fake, vectorizer_real, speller_real, stop_words_real, ps_real, top_words_real = bag_of_words.create_BOW_IG_env()

    dict_fake = dict()
    counter_fake = 0
    for word,freq, prob in top_words_fake:
        counter_fake += freq

    for word, freq, prob in top_words_fake:
        prob = freq / counter_fake
        dict_fake[word] = (freq, prob)
        print("fake word =", word, ", freq=",  freq, ", probability=", prob)

    dict_real = dict()
    counter_real = 0
    for word,freq, prob in top_words_real:
        counter_real += freq

    for word, freq, prob in top_words_real:
        prob = freq / counter_real
        dict_real[word] = (freq, prob)
        print("real word=", word, ", freq=", freq, ", probability=", prob)

    kl_divergence(top_words_fake, top_words_real, dict_fake, dict_real)

    return -1

def kl_divergence(top_words_fake, top_words_real, dict_fake, dict_real):

    print(top_words_fake)

    sum = 0
    for i in range(0, 20):
        fake_word = top_words_fake[i]
        print(fake_word)

        # check if fake word exists in "real" dictionary
        if fake_word[0] in dict_real:
            print("F(i): ", dict_fake.get(fake_word[0])[1])
            print("N(i): ", dict_real.get(fake_word[0])[1])
            sum += dict_fake.get(fake_word[0])[1] * np.log(dict_fake.get(fake_word[0])[1] / dict_real.get(fake_word[0])[1] )


    return -1



