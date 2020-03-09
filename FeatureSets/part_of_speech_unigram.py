import nltk
from nltk import RegexpTagger, UnigramTagger
from nltk.corpus import brown
from nltk import word_tokenize


def train_and_get_unigram_tagger():
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

    return UnigramTagger(train_text, backoff=regexp_tagger)


def get_unigrams_and_POS_tags_of_text(text, tagger):
    unigrams = word_tokenize(text)
    tagged_unigrams = tagger.tag(unigrams)

    res = " "
    for word, pos_tag in tagged_unigrams:
        res += word + " " + pos_tag + " "

    return res
