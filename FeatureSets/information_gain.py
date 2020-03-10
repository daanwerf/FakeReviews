from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from nltk.stem import PorterStemmer
from symspellpy.symspellpy import SymSpell, Verbosity
from nltk.corpus import stopwords
from FeatureExtraction import bag_of_words

def test_random_stuff(corpus, n):
    # corpus = [
    #     'This is the first document',
    #     'This is second document',
    #     'And this is the third one',
    #     'Is this the first document?',
    # ]

    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(corpus)
    # print("FROM IG CLASS TEST RANDOM STUFF")
    #
    # print("Vectorizer names:")
    # print(vectorizer.get_feature_names())
    #
    # print("X to Array:")
    # print(X.toarray())

    # vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2,2), stop_words='english')
    # X2 = vectorizer2.fit_transform(corpus)
    #
    # freqs = zip(vectorizer2.get_feature_names(), X2.sum(axis=0).tolist()[0])
    # print(sorted(freqs, key=lambda x: -x[1]))
    #
    # speller = SymSpell(max_dictionary_edit_distance=4)
    # dictionary_path = "../dictionaries/frequency_dictionary_en_82_765.txt"
    # speller.load_dictionary(dictionary_path, 0, 1)

    # print("new code IG:\n")
    # vec3 = CountVectorizer(stop_words='english').fit(corpus)
    # bow = vec3.transform(corpus)
    # sum_of_words = bow.sum(axis=0)
    # words_freq = [(word, sum_of_words[0, idx]) for word, idx in vec3.vocabulary_.items()]
    # words_freq = sorted(words_freq, key= lambda x: x[1], reverse=True)

   # return words_freq[:n]


    #newest code ----------


    vectorizer, speller, stop_words, ps, top_words  = bag_of_words.create_BOW_IG()

    counter = 0
    for word,freq in top_words:
        counter += freq
        print(word,freq)
        print("updated counter is: " , freq)


    return -1

