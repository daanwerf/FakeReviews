from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif

def test_random_stuff():
    corpus = [
        'This is the first document',
        'This is second document',
        'And this is the third one',
        'Is this the first document?',
    ]

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

   # res = dict(zip(vectorizer2.get_feature_names(), mutual_info_classif(X2)))

   # print(res)
    # print("Vectorizer 2 names:")
    # print(vectorizer2.get_feature_names())
    #
    # print("X2 to Array:")
    # print(X2.toarray())

    print("END IG ________________")


    return -1