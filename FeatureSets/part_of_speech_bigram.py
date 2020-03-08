from nltk import RegexpTagger, UnigramTagger, BigramTagger
from nltk.corpus import brown
from nltk import word_tokenize


def train_and_get_bigram_tagger():
    train_text = brown.tagged_sents()
    regexp_tagger = RegexpTagger(
                [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
                 (r'(The|the|A|a|An|an)$', 'AT'),   # articles
                 (r'.*able$', 'JJ'),                # adjectives
                 (r'.*ness$', 'NN'),                # nouns formed from adjectives
                 (r'.*ly$', 'RB'),                  # adverbs
                 (r'.*s$', 'NNS'),                  # plural nouns
                 (r'.*ing$', 'VBG'),                # gerunds
                 (r'.*ed$', 'VBD'),                 # past tense verbs
                 (r'.*', 'NN')                      # nouns (default)
            ])

    unigram_tagger = UnigramTagger(train_text, backoff=regexp_tagger)
    bigram_tagger = BigramTagger(train_text, backoff=unigram_tagger)

    return bigram_tagger


def get_bigrams_and_POS_tags_of_sentence(sentence, tagger):
    sentence = sentence.replace(",", "").replace(".", "").replace("-", " ").replace("=", " ")
    tagged_bigrams = tagger.tag(word_tokenize(sentence))

    res = ''
    previous_word, previous_tag = '', ''
    for word, pos_tag in tagged_bigrams:
        if previous_word == '':
            previous_word, previous_tag = word, pos_tag
        else:
            res += previous_word + "_" + word + ' ' + previous_tag + "_" + pos_tag + ' '
            previous_word, previous_tag = word, pos_tag

    return res
