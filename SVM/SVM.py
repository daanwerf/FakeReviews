from sklearn.model_selection import train_test_split
from HelperFunctions import yelp_dataset_functions as yelp
from FeatureExtraction import bag_of_words as bow
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_predict, cross_val_score
from FeatureSets import part_of_speech_unigram as unipos, part_of_speech_bigram as bipos

def make_preprocess_decision_dict(use_feature_set):
    preprocess = {
        'stop_words': False,
        'spell_checker': False,
        'stemmer': False,
        'unigram': False,
        'bigram': False,
        'unipos': False,
        'bipos': False
    }

    if use_feature_set == "unigram":
        preprocess['stop_words'] = True
        preprocess['spell_checker'] = True
        preprocess['stemmer'] = False
        preprocess['unigram'] = True
    elif use_feature_set == "bigram":
        preprocess['stop_words']: True
        preprocess['spell_checker'] = True
        preprocess['stemmer'] = False
        preprocess['bigram'] = True
    elif use_feature_set == "unipos":
        preprocess['stop_words']: True
        preprocess['spell_checker'] = True
        preprocess['stemmer'] = False
        preprocess['unipos'] = True
    elif use_feature_set == "bipos":
        preprocess['stop_words']: True
        preprocess['spell_checker'] = True
        preprocess['stemmer'] = False
        preprocess['bipos'] = True

    return preprocess


def execute_SVM_process(use_feature_set, create_new_Samples = False, save_features_and_labels = False, new_sample_amount = 1,
                        sample_size = 1000, use_sample = 0):
    if create_new_Samples:
        print("creating " + str(new_sample_amount) + " balanced samples of size " + str(sample_size))
        yelp.create_balanced_samples(new_sample_amount, sample_size)


    # Load dataset here.
    # data in the shape of an array of features. [[features_sample1],[features_sample2], etc]
    # target in the shape of [1 0 1 1 0 1] where 1 corresponds with the first sample.
    print("Reading sample file number " + str(use_sample))
    use_sample = 0
    sample_reader = yelp.get_balanced_sample_reader(use_sample)

    print("Initializing BOW environment for " + str(use_feature_set))
    preprocess = make_preprocess_decision_dict(use_feature_set)
    vectorizer, speller, stop_words, ps, tagger = bow.create_BOW_environment(preprocess, use_sample)
    print("Environment intialized!")

    print("Creating the feature and label arrays")
    X = np.zeros((sample_size, len(vectorizer.get_feature_names())))
    y = []

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
                [unipos.get_unigrams_and_POS_tags_of_text(
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

    if save_features_and_labels:
        print("saving features and labels")
        np.save("../Reviews/Yelp_Dataset/bagOfWords/sample_" + str(use_sample) + "_features", X)
        np.save("../Reviews/Yelp_Dataset/bagOfWords/sample_" + str(use_sample) + "_labels", y)
        print("finished saving, creating splits now")

    # Split dataset into training set and test set. X is input, Y is target
    # replace data and target with the correct data and target.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print("X: " + str(np.shape(X_train)))
    print("y: " + str(np.shape(y_train)))

    print("Finished split, starting learning process")

    # 5-fold cross validation, Linear Kernel
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

    print("Done learning, calculating metrics now")

    accuracies = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("5-fold accuracies: " + str(accuracies) + " average: " + str(sum(accuracies) / len(accuracies)))
    precisions = cross_val_score(clf, X, y, cv=5, scoring='precision')
    print("5-fold precisions: " + str(precisions) + " average: " + str(sum(precisions) / len(precisions)))
    recalls = cross_val_score(clf, X, y, cv=5, scoring='recall')
    print("5-fold recalls: " + str(recalls) + " average: " + str(sum(recalls) / len(recalls)))
    f1s = cross_val_score(clf, X, y, cv=5, scoring='f1')
    print("5-fold f1-scores: " + str(f1s) + " average: " + str(sum(f1s) / len(f1s)))


execute_SVM_process('bipos')
