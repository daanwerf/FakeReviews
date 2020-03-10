from nltk.stem import PorterStemmer
from symspellpy.symspellpy import SymSpell, Verbosity
from HelperFunctions import yelp_dataset_functions as yelp
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def make_sentence_array(reader, speller, stop_words, ps):
    sentences = []
    label, review_text = yelp.get_next_review_and_label(reader)
    while label != "-1":
        sentences.append(sanitize_sentence(review_text, speller, stop_words, ps))
        label, review_text = yelp.get_next_review_and_label(reader)

    return sentences

# Report: too high max distance -> too large change in words.
def get_words_from_symspell_lookup(sentence, speller):
    new_sentence = ""
    for word_spelling in sentence.split():
        lookup = speller.lookup(word_spelling, Verbosity.CLOSEST, max_edit_distance=2, ignore_token=r"\w+\d")
        if len(lookup) == 0:
            continue

        correction = str(lookup[0]).split(",")[0]
        if correction != word_spelling:
            new_sentence += " " + correction
        else:
            new_sentence += " " + word_spelling

    return new_sentence


# Report: 500 reviews: with sanitizing: 3001, without: 4459
def sanitize_sentence(sentence, speller, stop_words, ps):
    sentence = sentence.translate(str.maketrans('', '', r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~""")).lower()

    sentence = ' '.join([w for w in sentence.split() if len(w) > 2])
    sentence = ' '.join(s for s in sentence.split() if not any(c.isdigit() for c in s))
    sentence = ' '.join(s for s in sentence.split() if s not in stop_words)

    # Report: pyspellchecker too slow
    sentence = get_words_from_symspell_lookup(sentence, speller)

    stemmed_sentence = ""
    for word_to_stem in sentence.split():
        stemmed_sentence = stemmed_sentence + " " + ps.stem(word_to_stem)

    return stemmed_sentence


def create_BOW_environment():
    speller = SymSpell(max_dictionary_edit_distance=4)
    dictionary_path = "../dictionaries/frequency_dictionary_en_82_765.txt"
    speller.load_dictionary(dictionary_path, 0, 1)

    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    reader = yelp.get_balanced_sample_reader(0)

    sentences = make_sentence_array(reader, speller, stop_words, ps)
    reader.close()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)



    return vectorizer, speller, stop_words, ps


def create_BOW_IG():
    speller = SymSpell(max_dictionary_edit_distance=4)
    dictionary_path = "../dictionaries/frequency_dictionary_en_82_765.txt"
    speller.load_dictionary(dictionary_path, 0, 1)

    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    reader = yelp.get_balanced_sample_reader(0)

    sentences = make_sentence_array(reader, speller, stop_words, ps)
    reader.close()
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(sentences)

    sum_of_words = X.sum(axis=0)
    words_freq = [(word, sum_of_words[0, idx], 0) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key= lambda x: x[1], reverse=True)

    print(sum_of_words)
    print("IN BAG OF WORDS IN IG")

    return vectorizer, speller, stop_words, ps, words_freq

