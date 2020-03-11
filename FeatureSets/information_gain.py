from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from nltk.stem import PorterStemmer
from symspellpy.symspellpy import SymSpell, Verbosity
from nltk.corpus import stopwords
from FeatureSets import part_of_speech_bigram as bipos
from FeatureExtraction import bag_of_words as bow
from HelperFunctions import yelp_dataset_functions as yelp
from nltk import word_tokenize
from scipy.stats import entropy
import numpy as np
from itertools import islice


def kl_divergence(percentage, top_words_fake, top_words_real, dict_fake, dict_real):
    dict_kl_per_word = dict()

    # calculate KL(F||N)
    sum_fn = 0
    for word, freq, prob in top_words_fake:

        #Cannot divide by 0
        if word in dict_real:
            # print("for KL(F||N) -> F(i): ", dict_fake.get(word)[1])
            # print("for KL(F||N) -> N(i): ", dict_real.get(word)[1])
            kl_word = dict_fake.get(word)[1] * np.log(dict_fake.get(word)[1] / dict_real.get(word)[1])
            sum_fn += kl_word

            # add every word in fake  to kl dict with the corresponding kl_value

            dict_kl_per_word[word] = kl_word

    # print("total KL(F||N) for table 5 = ", sum_fn)

    # calculate KL(N||F)
    sum_nf = 0
    for word, freq, prob in top_words_real:

        #Cannot divide by 0
        if word in dict_fake:
            # print("for KL(N||F) -> F(i): ", dict_fake.get(word)[1])
            # print("for KL(N||F) -> N(i): ", dict_real.get(word)[1])
            kl_word = dict_real.get(word)[1] * np.log(dict_real.get(word)[1] / dict_fake.get(word)[1])
            sum_nf += kl_word

            #update to get the delta
            dict_kl_per_word[word] = dict_kl_per_word.get(word) - kl_word



    # print("total KL(N||F) for table 5 = ", sum_nf)
    # print("total delta KL for tale 5 = ", float(sum_fn - sum_nf))

    dict_kl_per_word = {k: v for k, v in sorted(dict_kl_per_word.items(), key=lambda x: abs(float(x[1])), reverse=True)}

    n_items = take(round(percentage * len(dict_kl_per_word)), dict_kl_per_word.items())

    # print("first 10 items based on they absolute delta kb: ", n_items)
    result = []
    for item in n_items:
        result.append(item[0])

    return result


def create_fake_sentences_IG(fake_sentences):
    vectorizer = CountVectorizer()

    X_fake = vectorizer.fit_transform(fake_sentences)

    sum_of_words_fake = X_fake.sum(axis=0)

    words_freq_fake = [(word, sum_of_words_fake[0, idx], 0.0) for word, idx in vectorizer.vocabulary_.items()]
    words_freq_fake = sorted(words_freq_fake, key= lambda x: x[1], reverse=True)

    return words_freq_fake


def create_real_sentences_IG(real_sentences):
    vectorizer = CountVectorizer()

    X_real = vectorizer.fit_transform(real_sentences)
    sum_of_words_real = X_real.sum(axis=0)

    words_freq_real = [(word, sum_of_words_real[0, idx], 0.0) for word, idx in vectorizer.vocabulary_.items()]
    words_freq_real = sorted(words_freq_real, key=lambda x: x[1], reverse=True)

    return words_freq_real


def find_words_real_and_fake(reader, speller, stop_words, ps, preprocess):
    real_set = []
    fake_set = []

    label, review = yelp.get_next_review_and_label(reader)

    while label != "-1":
        sanitized = word_tokenize(bow.sanitize_sentence(review, speller, stop_words, ps, preprocess))

        if label == "1":
            for word in sanitized:
                if word not in real_set:
                    real_set.append(word)

        if label == "0":
            for word in sanitized:
                if word not in fake_set:
                    fake_set.append(word)

        label, review = yelp.get_next_review_and_label(reader)

    return real_set, fake_set


def get_ig_features(percentage, reader, speller, stop_words, ps, preprocess):

    real_set, fake_set = find_words_real_and_fake(reader, speller, stop_words, ps, preprocess)

    top_words_real = create_real_sentences_IG(real_set)
    top_words_fake = create_fake_sentences_IG(fake_set)

    dict_fake = dict()
    counter_fake = 0
    for word, freq, prob in top_words_fake:
        counter_fake += freq

    for word, freq, prob in top_words_fake:
        prob = freq / counter_fake
        dict_fake[word] = (freq, prob)

    dict_real = dict()
    counter_real = 0
    for word, freq, prob in top_words_real:
        counter_real += freq

    for word, freq, prob in top_words_real:
        prob = freq / counter_real
        dict_real[word] = (freq, prob)

    result = kl_divergence(percentage, top_words_fake, top_words_real, dict_fake, dict_real)

    print(result)

    return result


#Return first n items of the iterable as a list
def take(n, iterable):
    return list(islice(iterable, n))



