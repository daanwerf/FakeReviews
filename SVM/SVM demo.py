from sklearn.model_selection import train_test_split
from HelperFunctions import yelp_dataset_functions as yelp
from FeatureExtraction import bag_of_words as bow
from FeatureSets import information_gain as IG
import numpy as np
from sklearn import svm, metrics, preprocessing
import nltk

save_features_and_labels = False
create_new_Samples = True
sample_amount = 1
sample_size = 100
if create_new_Samples:
    print("creating " + str(sample_amount) + " balanced samples of size " + str(sample_size))
    yelp.create_balanced_samples(sample_amount, sample_size)

# TEST IG STUFF

# corpus = [
#         'This is the first document',
#         'This is second document',
#         'And this is the third one',
#         'Is this the first document?',
#         'Everyone from New York city  knows you only go to Junior his for Cheese cake!!!! I would never eat the food there! Why they serve meals I have no idea because I do not know anyone personally who has ever bothered to eat meals there. The 4 stars is only for the cheese cake of course. But the cheese cakes are as much a part of Brooklyn New York history as Coney Island. They are the classic New York cheese cake. They used to be my favorite but like with most things that are firsts people always come and improve on them after wards. It is good that tourists can now eat Junior his in Manhattan and do not have to go all the way to Brooklyn to enjoy the cheese cake. Plus it seems not many people know this but all of Junior his cheese cakes at all of their locations come from the Maspeth warehouse where they are baked. So you are getting the same thing at the Manhattan locations as you would in Brooklyn. Plus the Brooklyn location was in the news for having mice. I have been to all of the locations.',
# 	    'Thank you, thank you, thank you to my colleague who took us on a hike down to Broome Street (from Hester) on our 45 minute lunch break, so we could pick up the Despana sandwiches we pre-ordered.  I walked into the place and I knew it was special.  I am sure I have passed it a bunch and did not even notice it but, ladies and gents, pay attention!  I guess in New York it is easy to leave stones unturned, but you better flip them over if you want to find some gems.   So, if you have not guessed, the theme of the place is:  Spain.  Myriad exotic and infused olive oils are out to sample and a comprehensive collection of Spanish cheeses are on display (including the awesome Roncal that I learned about in my Murray his Cheese class).  Every individual ingredient available at Espana is quality and together in something like..hmm... say the Quijote sandwich....they sing a fresh and lovely song.  My sandwich involved some type of pressed pork, but i reminded me of a cross between a pastrami and a salami.   It was different than any pork I have been familiar with.  Also,there was an excellent cheese (that  I now forget  the name of) and a sweet quince paste.  The bread was soft and it was coated in that layer of flour that lets you know it is fresh out the oven.   So this shop is a MUST try for a great quality sandwich and it is also excellent for any foodies who want to buy cooking ingredients, items for a cheese plate or gifts for a friend who is crafty in the kitchen.',
# 	    'I just ate the chicken fried pork with sausage gravy and biscuit.  It was even better then it sounds.',
# 	    'I love this place. I have tried several items on the menu and I am convinced that all of their food must be good. The restaurant is clean and trendy. The service has always been excellent. The wait staff, kitchen staff and manager are quite friendly and willing to talk and joke. I have enjoyed all of my dining experiences there.  I definitely see myself becoming a regular. Just beware, if you go on a Friday or Saturday night, you may have to wait in line!',
# 	    'Jesus. I do not think  I can give this place a bad review even if I tried. I have eaten here multiple times and I have never had a bad experience. The food and drinks are consistently great, and the service is friendly, if not occasionally harried (especially during brunch). It is definitely one of my go-to places in the Slope for great Latin food. It is not mom her cooking (her pernil will always beat what is served at restaurants) but it does come close.',
# ]

## retrieve the top n most common words
# topcommonwords = IG.test_random_stuff(corpus, 3)

#count total words
# counter = 0
# for word,freq in topcommonwords:
#     counter += freq
#     print(word, freq)




# Load dataset here.
# data in the shape of an array of features. [[features_sample1],[features_sample2], etc]
# target in the shape of [1 0 1 1 0 1] where 1 corresponds with the first sample.
use_sample = 0

sample_reader = yelp.get_balanced_sample_reader(use_sample)
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

print("y: " + str(y))

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
print("clf using linear kernel done")
rbfK = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
print("clf using rbf kernel done ")
sigm = svm.SVC(kernel='sigmoid', C=1).fit(X_train, y_train)
print("clf using sigmoid kernel done ")
# polyn = svm.SVC(kernel='polynomial', C=1).fit(X_train, y_train)
# print("clf using polynomial kernel done ")

print("-----Finished learning, making predictions now")


##### PREDICTION USING LINEAR KERNEL

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy linear kernel:", metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision linear kernel:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall linear kernel:", metrics.recall_score(y_test, y_pred))
print("--------------------")



##### PREDICTIONS USING RBF
r_pred = rbfK.predict(X_test)
print("Accuracy rbf kernel:", metrics.accuracy_score(y_test, r_pred))
print("Precision rbf kernel:", metrics.precision_score(y_test, r_pred))
print("Recall rbf kernel:", metrics.recall_score(y_test, r_pred))
print("--------------------")

##### PREDICTIONS USING SIGMOID
s_pred = sigm.predict(X_test)
print("Accuracy signmoid kernel:", metrics.accuracy_score(y_test, s_pred))
print("Precision sigmoid kernel:", metrics.precision_score(y_test, s_pred))
print("Recall sigmoid kernel:", metrics.recall_score(y_test, s_pred))
print("--------------------")

##### PREDICTIONS USING POLYNOMIMAL
# p_pred = polyn.predict(X_test)
# print("Accuracy polynomial kernel:", metrics.accuracy_score(y_test, p_pred))
# print("Precision polynomial kernel:", metrics.precision_score(y_test, p_pred))
# print("Recall polynomial kernel:", metrics.recall_score(y_test, p_pred))
