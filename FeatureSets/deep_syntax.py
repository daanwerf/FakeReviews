import re
from pickle import dump, load
from nltk.parse.corenlp import CoreNLPParser
from HelperFunctions import yelp_dataset_functions as yelp
from nltk import induce_pcfg, pchart, Nonterminal, word_tokenize
from FeatureExtraction import bag_of_words as bow
from FeatureSets import part_of_speech_bigram as bipos


# Report: struggles with reproducing "Deep syntax rules (obtained using Stanford Parser)" from paper
def create_grammar_of_sample(review_type, sample_id):
    # DONT FORGET TO RUN THE STANFORD CORENLP SERVER BY RUNNING THIS JAVA COMMAND IN THE ROOT FOLDER:
    # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9500 -timeout 30000

    if review_type == 'regular':
        reader = yelp.get_regular_balanced_sample_reader(sample_id)
    elif review_type == '45stars':
        reader = yelp.get_45stars_balanced_sample_reader(sample_id)
    elif review_type == '12stars':
        reader = yelp.get_12stars_balanced_sample_reader(sample_id)

    parser = CoreNLPParser(url='http://localhost:9500')
    productions = []

    label, review = yelp.get_next_review_and_label(reader)

    while label != "-1":
        for sentence in re.split(r"[.!?]", review):
            try:
                tree = next(parser.raw_parse(sentence))

                # Optimize by creating Chomsky normal form
                tree.collapse_unary(collapsePOS=False)
                tree.chomsky_normal_form(horzMarkov=2)
                productions += tree.productions()

            except StopIteration:
                # End of review reached
                break

            label, review = yelp.get_next_review_and_label(reader)

    S = Nonterminal('S')
    grammar = induce_pcfg(S, productions)

    output = open('../grammars/sample_' + review_type + "_" + str(sample_id), 'wb')
    dump(grammar, output, -1)
    output.close()


def load_sample_grammar(review_type, sample_id):
    input_grammar = open('../grammars/sample_' + review_type + "_" + str(sample_id), 'rb')
    grammar = load(input_grammar)
    input_grammar.close()

    return grammar


def get_bigram_and_deep_syntax_feature(review, speller, stop_words, ps, preprocess):
    res = ""
    productions = []

    parser = CoreNLPParser(url='http://localhost:9500')

    for sentence in re.split(r"[.!?]", review):
        try:
            tree = next(parser.raw_parse(sentence))

            # Optimize by creating Chomsky normal form
            tree.collapse_unary(collapsePOS=False)
            tree.chomsky_normal_form(horzMarkov=2)
            productions += tree.productions()

        except StopIteration:
            # End of review reached
            break

    S = Nonterminal('S')
    grammar = induce_pcfg(S, productions)

    count = 0
    for line in str(grammar).split("\n"):
        if count == 0:
            count += 1
            continue
        elif "'" in line:
            res += re.sub(r"[(->) `\'\"\[\d\]]", "", line) + " "

    res += bipos.get_bigrams_and_unigrams_of_sentence(
        bow.sanitize_sentence(review, speller, stop_words, ps, preprocess))

    return res





def parse_stuff(review_text, grammar):
    # Parse a sentence by induced grammar
    example_sentence = re.split(r"[.!?]", review_text)[0]

    insideChartParser = pchart.InsideChartParser(grammar)
    insideChartParser.trace(3)

    print(word_tokenize(example_sentence))

    leng = 0
    for parse in insideChartParser.parse(word_tokenize(example_sentence)):
        leng += 1

    print(leng)
