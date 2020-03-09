from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif

def test_random_stuff(corpus, n):
    # corpus = [
    #     'This is the first document',
    #     'This is second document',
    #     'And this is the third one',
    #     'Is this the first document?',
    # ]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print("FROM IG CLASS TEST RANDOM STUFF")

    print("Vectorizer names:")
    print(vectorizer.get_feature_names())

    print("X to Array:")
    print(X.toarray())

    vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2,2), stop_words='english')
    X2 = vectorizer2.fit_transform(corpus)

    freqs = zip(vectorizer2.get_feature_names(), X2.sum(axis=0).tolist()[0])
    print(sorted(freqs, key=lambda x: -x[1]))

    print("new code IG:\n")
    vec3 =CountVectorizer().fit(corpus)
    bow = vec3.transform(corpus)
    sum_of_words = bow.sum(axis=0)
    words_freq = [(word, sum_of_words[0, idx]) for word, idx in vec3.vocabulary_.items()]
    words_freq = sorted(words_freq, key= lambda x: x[1], reverse=True)

    print("END IG ________________")


    return words_freq[:n]

   # res = dict(zip(vectorizer2.get_feature_names(), mutual_info_classif(X2)))

   # print(res)
    # print("Vectorizer 2 names:")
    # print(vectorizer2.get_feature_names())
    #
    # print("X2 to Array:")
    # print(X2.toarray())


    # return -1