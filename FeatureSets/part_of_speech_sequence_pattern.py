from nltk import word_tokenize
from FeatureSets import part_of_speech_unigram as unipos, part_of_speech_bigram as bipos
from FeatureExtraction import bag_of_words as bow

import re


def mine_part_of_speech_pattern(D, T, minsup, minadherence, max_length):
    C = []

    C.append({})
    for d in D:
        for tag in d.split(" "):
            if tag == "" or tag == " ":
                continue
            if tag in C[0]:
                C[0][tag] += 1
            else:
                C[0][tag] = 1

    F = []
    n = len(D)

    F.append([])
    for f in C[0].keys():
        if (C[0][f] / n) >= minsup:
            F[0].append(f)

    SP = ""
    for f in F[0]:
        SP += f + " "

    for k in range(1, max_length):
        C = candidate_gen(F, T, C, k)

        for d in D:
            for c in C[k].keys():
                if contains_sequence(d, c):
                    C[k][c] += 1

        F.append([])
        for c in C[k].keys():
            if (C[k][c] / n) >= minsup:
                F[k].append(c)

        for f in F[k]:
            if fairSCP(f, D) >= minadherence:
                SP += f.replace(" ", "_")
                SP += " "

    return SP


def candidate_gen(F, T, C, k):
    C.append({})

    for c in F[k-1]:
        for t in T:
            new_c = add_suffix(c, t)
            C[k][new_c] = 0

    return C


def add_suffix(c, t):
    return str(c) + " " + str(t)


def fairSCP(search_sequence, total_sequence):
    try:
        for d in total_sequence:
            result = 0

            n = len(search_sequence)
            for i in range(n):
                result += (probability_of_sequence(search_sequence[:i], d) * probability_of_sequence(search_sequence[i:], d))

            result = result * (1/(n-1))

            return (probability_of_sequence(search_sequence, d) ** 2) / result
    except ZeroDivisionError:
        return 0


def probability_of_sequence(search_sequence, total_sequence):
    search_sequence = word_tokenize(search_sequence)
    if type(total_sequence) == str:
        total_sequence = word_tokenize(total_sequence)
    count_found = 0
    count_sequence = 0

    for i in range(len(total_sequence)):
        found_sequence = []
        for j in range(len(search_sequence)):
            try:
                word2 = total_sequence[i+j]
                found_sequence.append(word2)
            except IndexError:
                break

        if len(found_sequence) == len(search_sequence):
            count_found += 1
            if word_lists_equal(found_sequence, search_sequence):
                count_sequence += 1

    return count_sequence/count_found


def word_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    else:
        for word in list1:
            if not word in list2:
                return False

    return True


def contains_sequence(total_sequence, search_sequence):
    total_sequence = word_tokenize(total_sequence)
    search_sequence = word_tokenize(search_sequence)

    for i in range(len(total_sequence)):
        found_sequence = []
        for j in range(len(search_sequence)):
            try:
                word2 = total_sequence[i+j]
                found_sequence.append(word2)
            except IndexError:
                break

        if len(found_sequence) == len(search_sequence):
            if word_lists_equal(found_sequence, search_sequence):
                return True

    return False


def get_total_and_tagset_from_sequence(D):
    result = []
    for d in D:
        for tag in d.split(" "):
            if tag != "" and tag not in result:
                result.append(tag)

    return result



def get_POS_sequence(review, tagger, speller, stop_words, ps, preprocess):
    review = bow.sanitize_sentence(review, speller, stop_words, ps, preprocess)
    sentences = re.split(r"[.!?]", review)
    sentences = sentences[:len(sentences)-1]

    D = [unipos.get_unigram_POS_tags_of_text(d, tagger) for d in sentences]
    T = get_total_and_tagset_from_sequence(D)

    minsup = 0.6
    minadherence = 0.4
    max_length = 4

    result = mine_part_of_speech_pattern(D, T, minsup, minadherence, max_length) + \
        re.sub(r"[.!?]", "", bipos.get_bigrams_and_unigrams_of_sentence(review))

    return result
