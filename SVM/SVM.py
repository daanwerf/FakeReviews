from sklearn.model_selection import train_test_split
from HelperFunctions import yelp_dataset_functions as yelp
from FeatureExtraction import bag_of_words as bow
import numpy as np
from sklearn import svm, metrics, preprocessing


def execute_SVM_process(use_feature_set, create_new_Samples = False, save_features_and_labels = False, sample_amount = 1,
                        sample_size = 1000, use_sample = 0):
    if create_new_Samples:
        print("creating " + str(sample_amount) + " balanced samples of size " + str(sample_size))
        yelp.create_balanced_samples(sample_amount, sample_size)

    # Load dataset here.
    # data in the shape of an array of features. [[features_sample1],[features_sample2], etc]
    # target in the shape of [1 0 1 1 0 1] where 1 corresponds with the first sample.
    print("Reading sample number " + str(use_sample))
    use_sample = 0
    sample_reader = yelp.get_balanced_sample_reader(use_sample)

    print("Creating feature sets")






    print("Done creating samples, initializing BOW environment")
    vectorizer, speller, stop_words, ps = bow.create_BOW_environment()
    print("Environment intialized!")

    print("Creating the feature and label arrays")
    X = np.zeros((sample_size, len(vectorizer.get_feature_names())))
    print("X intended shape: " + str(np.shape(X)))
    y = []

    label, review = yelp.get_next_review_and_label(sample_reader)
    counter = 0

    # Report: matrix addition waaaaaaaayyy faster than appending
    while label != "-1":
        y.append(int(label))
        X[counter] = X[counter] + vectorizer.transform([bow.sanitize_sentence(review, speller, stop_words, ps)])
        label, review = yelp.get_next_review_and_label(sample_reader)
        print("Progress: " + str((counter/sample_size) * 100))
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

    # Linear Kernel
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

    print("Finished learning, making predictions now")

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))
