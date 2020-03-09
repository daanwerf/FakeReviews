from nltk.stem import PorterStemmer
from symspellpy.symspellpy import SymSpell, Verbosity
from HelperFunctions import yelp_dataset_functions as yelp
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from FeatureSets import part_of_speech_bigram as bipos, part_of_speech_unigram as unipos

# Report: struggles with getting good bipos features
def make_sentence_array(reader, speller, stop_words, ps, tagger, preprocess):
    sentences = []
    label, review_text = yelp.get_next_review_and_label(reader)
    while label != "-1":
        sanitized_sentence = sanitize_sentence(review_text, speller, stop_words, ps, preprocess)

        if preprocess['unigram']:
            sentences.append(sanitized_sentence)
        elif preprocess['bigram']:
            sentences.append(bipos.get_bigrams_and_unigrams_of_sentence(sanitized_sentence))
        elif preprocess['bipos']:
            sentences.append(bipos.get_bigrams_and_POS_tags_of_sentence(sanitized_sentence, tagger))
        elif preprocess['unipos']:
            sentences.append(unipos.get_unigrams_and_POS_tags_of_text(sanitized_sentence, tagger))

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
def sanitize_sentence(sentence, speller, stop_words, ps, preprocess):
    sentence = sentence.translate(str.maketrans('', '', r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~""")).lower()

    sentence = ' '.join([w for w in sentence.split() if len(w) > 1])
    sentence = ' '.join(s for s in sentence.split() if not any(c.isdigit() for c in s))

    if preprocess['stop_words']:
        sentence = ' '.join(s for s in sentence.split() if s not in stop_words)

    # Report: pyspellchecker too slow
    if preprocess['spell_checker']:
        sentence = get_words_from_symspell_lookup(sentence, speller)

    if preprocess['stemmer']:
        stemmed_sentence = ""
        for word_to_stem in sentence.split():
            stemmed_sentence = stemmed_sentence + " " + ps.stem(word_to_stem)
        sentence = stemmed_sentence

    return sentence


def create_BOW_environment(preprocess, use_sample):
    speller = None
    if preprocess['spell_checker']:
        print("Loading speller dictionary")
        speller = SymSpell()
        dictionary_path = "../dictionaries/frequency_dictionary_en_82_765.txt"
        speller.load_dictionary(dictionary_path, 0, 1)

    ps = None
    if preprocess['stemmer']:
        print("Loading stemmer")
        ps = PorterStemmer()

    stop_words = None
    if preprocess['stop_words']:
        print("Loading stop words")
        stop_words = set(stopwords.words('english'))

    tagger = None
    if preprocess['bipos']:
        print("Loading and training bipos tagger")
        tagger = bipos.train_and_get_bigram_tagger()
    elif preprocess['unipos']:
        print("Loading and training unipos tagger")
        tagger = unipos.train_and_get_unigram_tagger()

    reader = yelp.get_regular_balanced_sample_reader(use_sample)

    sentences = make_sentence_array(reader, speller, stop_words, ps, tagger, preprocess)
    reader.close()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    return vectorizer, speller, stop_words, ps, tagger
