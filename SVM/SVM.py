from sklearn.model_selection import train_test_split
from HelperFunctions import yelp_dataset_functions as yelp
from FeatureExtraction import bag_of_words as bow
import numpy as np
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, KFold
from FeatureSets import part_of_speech_unigram as unipos, part_of_speech_bigram as bipos
from FeatureSets import deep_syntax as ds, part_of_speech_sequence_pattern as posseq


def make_preprocess_decision_dict(use_feature_set):
    preprocess = {
        'stop_words': False,
        'spell_checker': False,
        'stemmer': False,
        'unigram': False,
        'bigram': False,
        'unipos': False,
        'bipos': False,
        'deep': False,
        'posseq': False,
        'ig1%': False,
        'ig2%': False,
    }

    if use_feature_set == "unigram":
        preprocess['stop_words'] = True
        preprocess['spell_checker'] = True
        preprocess['stemmer'] = False
        preprocess['unigram'] = True
    elif use_feature_set == "bigram":
        preprocess['stop_words'] = True
        preprocess['spell_checker'] = True
        preprocess['stemmer'] = False
        preprocess['bigram'] = True
    elif use_feature_set == "unipos":
        preprocess['stop_words'] = True
        preprocess['spell_checker'] = True
        preprocess['stemmer'] = False
        preprocess['unipos'] = True
    elif use_feature_set == "bipos":
        preprocess['stop_words'] = True
        preprocess['spell_checker'] = True
        preprocess['stemmer'] = False
        preprocess['bipos'] = True
    elif use_feature_set == "deep":
        preprocess['stop_words'] = True
        preprocess['spell_checker'] = True
        preprocess['stemmer'] = False
        preprocess['deep'] = True
    elif use_feature_set == "posseq":
        preprocess['stop_words'] = True
        preprocess['spell_checker'] = False
        preprocess['stemmer'] = False
        preprocess['posseq'] = True
    elif use_feature_set == "ig1%":
        preprocess['stop_words'] = True
        preprocess['spell_checker'] = True
        preprocess['stemmer'] = False
        preprocess['ig1%'] = True
    elif use_feature_set == "ig2%":
        preprocess['stop_words'] = True
        preprocess['spell_checker'] = True
        preprocess['stemmer'] = False
        preprocess['ig2%'] = True

    return preprocess

def execute_SVM_process(review_type, use_feature_set, create_new_samples=False, new_sample_amount=1,
                        sample_size=800, use_sample=0):
    if create_new_samples:
        print("creating " + str(new_sample_amount) + " " + review_type + " balanced samples of size " + str(sample_size))
        yelp.create_balanced_samples(review_type, new_sample_amount, sample_size)

    print("Reading " + review_type + " sample file number " + str(use_sample))
    use_sample = 0
    if review_type == 'regular':
        sample_reader = yelp.get_regular_balanced_sample_reader(use_sample)
    elif review_type == '45stars':
        sample_reader = yelp.get_45stars_balanced_sample_reader(use_sample)
    elif review_type == '12stars':
        sample_reader = yelp.get_12stars_balanced_sample_reader(use_sample)

    print("Initializing BOW environment for " + str(use_feature_set))
    preprocess = make_preprocess_decision_dict(use_feature_set)
    vectorizer, speller, stop_words, ps, tagger = bow.create_BOW_environment(preprocess, use_sample, sample_reader)

    print("Creating the feature and label arrays")
    X = np.zeros((sample_size, len(vectorizer.get_feature_names())))
    y = []

    # Get a new reader
    if review_type == 'regular':
        sample_reader = yelp.get_regular_balanced_sample_reader(use_sample)
    elif review_type == '45stars':
        sample_reader = yelp.get_45stars_balanced_sample_reader(use_sample)
    elif review_type == '12stars':
        sample_reader = yelp.get_12stars_balanced_sample_reader(use_sample)

    label, review = yelp.get_next_review_and_label(sample_reader)
    counter = 0

    # Report: matrix addition waaaaaaaayyy faster than appending
    while label != "-1":
        y.append(int(label))

        if preprocess['unigram']:
            X[counter] = X[counter] + vectorizer.transform(
                [bow.sanitize_sentence(review, speller, stop_words, ps, preprocess)])
            label, review = yelp.get_next_review_and_label(sample_reader)
            print("Progress: " + str((counter/sample_size) * 100) + "%")
            counter += 1
        elif preprocess['bigram']:
            X[counter] = X[counter] + vectorizer.transform(
                [bipos.get_bigrams_and_unigrams_of_sentence(
                    bow.sanitize_sentence(review, speller, stop_words, ps, preprocess))])
            label, review = yelp.get_next_review_and_label(sample_reader)
            print("Progress: " + str((counter / sample_size) * 100) + "%")
            counter += 1
        elif preprocess['unipos']:
            X[counter] = X[counter] + vectorizer.transform(
                [unipos.get_unigram_POS_tags_of_text(
                    bow.sanitize_sentence(review, speller, stop_words, ps, preprocess), tagger)])
            label, review = yelp.get_next_review_and_label(sample_reader)
            print("Progress: " + str((counter / sample_size) * 100) + "%")
            counter += 1
        elif preprocess['bipos']:
            X[counter] = X[counter] + vectorizer.transform(
                [bipos.get_bigrams_and_POS_tags_of_sentence(
                    bow.sanitize_sentence(review, speller, stop_words, ps, preprocess), tagger)])
            label, review = yelp.get_next_review_and_label(sample_reader)
            print("Progress: " + str((counter / sample_size) * 100) + "%")
            counter += 1
        elif preprocess['deep']:
            X[counter] = X[counter] + vectorizer.transform(
                [ds.get_bigram_and_deep_syntax_feature(review, speller, stop_words, ps, preprocess)]
            )
            label, review = yelp.get_next_review_and_label(sample_reader)
            print("Progress: " + str((counter / sample_size) * 100) + "%")
            counter += 1
        elif preprocess['posseq']:
            X[counter] = X[counter] + vectorizer.transform(
                [posseq.get_POS_sequence(review, tagger, speller, stop_words, ps, preprocess)]
            )
            label, review = yelp.get_next_review_and_label(sample_reader)
            print("Progress: " + str((counter / sample_size) * 100) + "%")
            counter += 1
        elif preprocess['ig1%'] or preprocess['ig2%']:
            X[counter] = X[counter] + vectorizer.transform(
                [bow.sanitize_sentence(review, speller, stop_words, ps, preprocess)]
            )
            label, review = yelp.get_next_review_and_label(sample_reader)
            print("Progress: " + str((counter / sample_size) * 100) + "%")
            counter += 1

    sample_reader.close()

    y = np.asarray(y)
    print(X.shape)
    print(y.shape)

    # 5-fold cross validation, Linear Kernel
    print("Starting 5-fold cross-validation learning")

    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for train_index, test_index in kf.split(X):
        data_train = X[train_index]
        target_train = y[train_index]

        data_test = X[test_index]
        target_test = y[test_index]

        clf = svm.SVC(kernel='linear', C=0.65)
        clf.fit(data_train, target_train)

        predictions = clf.predict(data_test)

        # accuracy for the current fold only
        accuracy = accuracy_score(target_test, predictions)
        accuracies.append(accuracy)

        # precision for the current fold only
        precision = precision_score(target_test, predictions)
        precisions.append(precision)

        # recall for the current fold only
        recall = recall_score(target_test, predictions)
        recalls.append(recall)

        # f1 for the current fold only
        f1 = f1_score(target_test, predictions)
        f1s.append(f1)

    average_accuracy = np.mean(accuracies)
    print("Accuracy: " + str(average_accuracy))

    average_precision = np.mean(precisions)
    print("Precision: " + str(average_precision))

    average_recall = np.mean(recalls)
    print("Recall: " + str(average_recall))

    average_f1 = np.mean(f1s)
    print("f1: " + str(average_f1))


execute_SVM_process('12stars', 'ig2%', create_new_samples=False)
