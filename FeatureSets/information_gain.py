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

    #print(top_words_fake)

    sum_fn = 0
    for word, freq, prob in top_words_fake:

        #calculate KL(F||N)
        if word in dict_real:
            print("for KL(F||N) -> F(i): ", dict_fake.get(word)[1])
            print("for KL(F||N) -> N(i): ", dict_real.get(word)[1])
            sum_fn +=  dict_fake.get(word)[1] * np.log(dict_fake.get(word)[1] / dict_real.get(word)[1])
    print("total KL(F||N) = ", sum_fn)

    sum_nf = 0
    for word, freq, prob in top_words_real:

        #calculate KL(N||F)
        if word in dict_fake:
            print("for KL(N||F) -> F(i): ", dict_fake.get(word)[1])
            print("for KL(N||F) -> N(i): ", dict_real.get(word)[1])
            sum_nf +=  dict_real.get(word)[1] * np.log(dict_real.get(word)[1] / dict_fake.get(word)[1])

    print("total KL(N||F) = ", sum_nf)




    #
    # for i in range(0, 20): # change to for loop over ALL WORDS -> word, freq, prob in blablal.
    #
    #     # calculate KL(F||N) --> dict_fake.get(word)[1]
    #     # dict = {"word", (freq, prob) }
    #     # dict.get("food")[1] => prob van food
    #     fake_word = top_words_fake[i]
    #     print(fake_word)
    #
    #     # check if fake word exists in "real" dictionary
    #     if fake_word[0] in dict_real:
    #         print("F(i): ", dict_fake.get(fake_word[0])[1])
    #         print("N(i): ", dict_real.get(fake_word[0])[1])
    #         sum += dict_fake.get(fake_word[0])[1] * np.log(dict_fake.get(fake_word[0])[1] / dict_real.get(fake_word[0])[1] )

        # do same shit for KL(N||F)


        # take delta

        # sort on delta

        # take top n based


    return -1



