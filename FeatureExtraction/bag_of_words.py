from nltk.stem import PorterStemmer
from symspellpy.symspellpy import SymSpell, Verbosity
from HelperFunctions import yelp_dataset_functions as yelp
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def make_sentence_array(reader, speller, stop_words, ps):
    sentences_total = []
    sentences_real = []
    sentences_fake = []
    label, review_text = yelp.get_next_review_and_label(reader)
    while label == "1":
        sentences_real.append(sanitize_sentence(review_text, speller, stop_words, ps))
        label, review_text = yelp.get_next_review_and_label(reader)
    while label == "0":
        sentences_fake.append(sanitize_sentence(review_text, speller, stop_words,ps))
        label, review_text = yelp.get_next_review_and_label(reader)

    sentences_total = sentences_real + sentences_fake
    return sentences_total, sentences_fake , sentences_real


# return all the fake reviews for information gain
# def make_sentence_array_fake(reader, speller, stop_words, ps):
#     sentences = []
#     label, review_text = yelp.get_next_review_and_label(reader)
#     print(label)
#     while label == "1":
#         label, review_text = yelp.get_next_review_and_label(reader)
#     while label == "0":
#         sentences.append(sanitize_sentence(review_text, speller, stop_words, ps))
#         label, review_text = yelp.get_next_review_and_label(reader)
#
#     return sentences

#return all the real review for information gain
# def make_sentence_array_real(reader, speller, stop_words, ps):
#     sentences = []
#     label, review_text = yelp.get_next_review_and_label(reader)
#     while label == "1":
#         sentences.append(sanitize_sentence(review_text, speller, stop_words, ps))
#         label, review_text = yelp.get_next_review_and_label(reader)
#
#     return sentences


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

    sentences_total, sentences_fake, sentences_real = make_sentence_array(reader, speller, stop_words, ps)
    reader.close()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences_total)



    return vectorizer, speller, stop_words, ps


def create_BOW_IG_env():
    speller = SymSpell(max_dictionary_edit_distance=4)
    dictionary_path = "../dictionaries/frequency_dictionary_en_82_765.txt"
    speller.load_dictionary(dictionary_path, 0, 1)

    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    reader = yelp.get_balanced_sample_reader(0)

    total_sentences, fake_sentences, real_sentences = make_sentence_array(reader, speller, stop_words, ps)

    vectorizer_fake, speller_fake, stop_words_fake, ps_fake, words_freq_fake = create_fake_sentences_IG(reader, speller, stop_words, ps, fake_sentences)
    vectorizer_real, speller_real, stop_words_real, ps_real, words_freq_real = create_real_sentences_IG(reader, speller, stop_words, ps, real_sentences)

    reader.close()

    return vectorizer_fake, speller_fake, stop_words_fake, ps_fake, words_freq_fake,vectorizer_real, speller_real, stop_words_real, ps_real, words_freq_real



def create_fake_sentences_IG(reader, speller, stop_words, ps, fake_sentences):

    vectorizer = CountVectorizer()

    X_fake = vectorizer.fit_transform(fake_sentences)

    sum_of_words_fake = X_fake.sum(axis=0)

    words_freq_fake = [(word, sum_of_words_fake[0, idx], 0) for word, idx in vectorizer.vocabulary_.items()]
    words_freq_fake = sorted(words_freq_fake, key= lambda x: x[1], reverse=True)

    return vectorizer, speller, stop_words, ps, words_freq_fake

def create_real_sentences_IG(reader, speller, stop_words, ps, real_sentences):

    vectorizer = CountVectorizer()

    X_real = vectorizer.fit_transform(real_sentences)
    sum_of_words_real = X_real.sum(axis=0)

    words_freq_real = [(word, sum_of_words_real[0, idx], 0) for word, idx in vectorizer.vocabulary_.items()]
    words_freq_real = sorted(words_freq_real, key=lambda x: x[1], reverse=True)


    return vectorizer, speller, stop_words, ps, words_freq_real
