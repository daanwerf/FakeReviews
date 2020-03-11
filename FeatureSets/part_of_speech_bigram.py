from nltk import RegexpTagger, UnigramTagger, BigramTagger
from nltk.corpus import brown
from nltk import word_tokenize
from nltk import PCFG
from nltk.corpus import treebank
from nltk import Nonterminal
from nltk import treetransforms
from nltk import induce_pcfg
from nltk.parse import pchart
from pickle import dump, load


def train_and_save_bigram_tagger():
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

    output = open('../taggers/bigram_tagger.pkl', 'wb')
    dump(bigram_tagger, output, -1)
    output.close()


def load_bigram_tagger():
    input_tagger = open('../taggers/bigram_tagger.pkl', 'rb')
    tagger = load(input_tagger)
    input_tagger.close()

    return tagger


def get_bigrams_and_unigrams_of_sentence(sentence):
    res = ''
    previous_word = ''
    for word in sentence.split():
        if previous_word == '':
            previous_word = word
        else:
            res += previous_word + " " + previous_word + "_" + word + " "
            previous_word = word

    return res

# Report: other forms of bigram POS features scored a lower accuracy
def get_bigrams_and_POS_tags_of_sentence(sentence, tagger):
    tagged_bigrams = tagger.tag(word_tokenize(sentence))

    res = ''
    previous_word, previous_tag = '', ''
    for word, pos_tag in tagged_bigrams:
        if previous_word == '':
            previous_word, previous_tag = word, pos_tag
        else:
            res += previous_word + " " + previous_word + "_" + word + " " + previous_tag + "_" + pos_tag + ' '
            previous_word, previous_tag = word, pos_tag

    return res
