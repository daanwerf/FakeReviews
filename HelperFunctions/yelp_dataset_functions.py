from unicodedata import normalize
from re import sub
from HelperFunctions import undersampling as us
from nltk import word_tokenize


def create_labeled_review_dataset_from_source():
    metadata_file = open('../Reviews/Yelp_Dataset/yelpzip/metadata', newline='', encoding="utf8")
    review_file = open('../Reviews/Yelp_Dataset/yelpzip/reviewContent', newline='', encoding="utf8")

    outfile = open("../Reviews/Yelp_Dataset/allStarReviews/labeled_review.txt", "w", encoding="utf8")

    metadata_line = metadata_file.readline().split("\t")
    metadata_id = metadata_line[0]
    metadata_label = metadata_line[3]
    if metadata_label == "-1":
        metadata_label = "0"

    review_line = review_file.readline().split("\t")
    review_id = review_line[0]
    review_text = review_line[3]

    total_lines = 0
    while metadata_id == review_id:
        outfile.write(metadata_label + "\t" + review_text)
        total_lines += 1

        try:
            metadata_line = metadata_file.readline().split("\t")
            metadata_id = metadata_line[0]
            metadata_label = metadata_line[3]
            if metadata_label == "-1":
                metadata_label = "0"

            review_line = review_file.readline().split("\t")
            review_id = review_line[0]
            review_text = review_line[3]

        except IndexError:
            break
            # End of file

    metadata_file.close()
    review_file.close()
    outfile.close()
    print(total_lines)



def create_labeled_fake_and_true_review_dataset_from_source():
    metadata_file = open('../Reviews/Yelp_Dataset/yelpzip/metadata', newline='', encoding="utf8")
    review_file = open('../Reviews/Yelp_Dataset/yelpzip/reviewContent', newline='', encoding="utf8")

    fake_outfile = open("../Reviews/Yelp_Dataset/allStarReviews/labeled_reviews_fake.txt", "w", encoding="utf8")
    true_outfile = open("../Reviews/Yelp_Dataset/allStarReviews/labeled_reviews_true.txt", "w", encoding="utf8")

    metadata_line = metadata_file.readline().split("\t")
    metadata_id = metadata_line[0]
    metadata_label = metadata_line[3]
    if metadata_label == "-1":
        metadata_label = "0"

    review_line = review_file.readline().split("\t")
    review_id = review_line[0]
    review_text = review_line[3]

    fake_amount = 0
    true_amount = 0
    total_amount = 0

    while metadata_id == review_id:
        if metadata_label == "0":
            fake_outfile.write(metadata_label + "\t" + review_text)
            fake_amount += 1
        elif metadata_label == "1":
            true_outfile.write(metadata_label + "\t" + review_text)
            true_amount += 1

        total_amount += 1
        try:
            metadata_line = metadata_file.readline().split("\t")
            metadata_id = metadata_line[0]
            metadata_label = metadata_line[3]
            if metadata_label == "-1":
                metadata_label = "0"

            review_line = review_file.readline().split("\t")
            review_id = review_line[0]
            review_text = review_line[3]

        except IndexError:
            break
            # End of file

    metadata_file.close()
    review_file.close()
    fake_outfile.close()
    true_outfile.close()

    print(fake_amount)
    print(true_amount)
    print(total_amount)


def create_labeled_fake_and_true_review_dataset_from_validation_set():
    validation_set_reader = open('../Reviews/Yelp_Dataset/machineLearningSets/validation_set.txt', newline='', encoding="utf8")

    fake_outfile = open("../Reviews/Yelp_Dataset/machineLearningSets/labeled_reviews_validation_fake.txt", "w", encoding="utf8")
    true_outfile = open("../Reviews/Yelp_Dataset/machineLearningSets/labeled_reviews_validation_true.txt", "w", encoding="utf8")

    line = validation_set_reader.readline().split("\t")
    label = line[0]
    review = line[1]

    fake_amount = 0
    true_amount = 0
    total_amount = 0

    while label != '':
        if label == "0":
            fake_outfile.write(label + "\t" + review.replace("\n", ""))
            fake_amount += 1
        elif label == "1":
            true_outfile.write(label + "\t" + review.replace("\n", ""))
            true_amount += 1

        total_amount += 1

        line = validation_set_reader.readline().split("\t")
        label = line[0]
        if line[0] == '':
            break
        review = line[1]



    validation_set_reader.close()
    fake_outfile.close()
    true_outfile.close()

    print(fake_amount)
    print(true_amount)
    print(total_amount)


def create_4_and_5_star_labeled_review_dataset_from_source():
    metadata_file = open('../Reviews/Yelp_Dataset/yelpzip/metadata', newline='', encoding="utf8")
    review_file = open('../Reviews/Yelp_Dataset/yelpzip/reviewContent', newline='', encoding="utf8")

    outfile = open("../Reviews/Yelp_Dataset/fourAndFiveStarReviews/labeled_review.txt", "w", encoding="utf8")

    metadata_line = metadata_file.readline().split("\t")
    metadata_id = metadata_line[0]
    metadata_score = metadata_line[2]
    metadata_label = metadata_line[3]
    if metadata_label == "-1":
        metadata_label = "0"

    review_line = review_file.readline().split("\t")
    review_id = review_line[0]
    review_text = review_line[3]

    total_lines = 0
    while metadata_id == review_id:
        if metadata_score == "4.0" or metadata_score == "5.0":
            outfile.write(metadata_label + "\t" + review_text)
            total_lines += 1

        try:
            metadata_line = metadata_file.readline().split("\t")
            metadata_id = metadata_line[0]
            metadata_score = metadata_line[2]
            metadata_label = metadata_line[3]
            if metadata_label == "-1":
                metadata_label = "0"

            review_line = review_file.readline().split("\t")
            review_id = review_line[0]
            review_text = review_line[3]

        except IndexError:
            break
            # End of file

    metadata_file.close()
    review_file.close()
    outfile.close()
    print(total_lines)


def create_4_and_5_star_labeled_fake_and_true_review_dataset_from_source():
    metadata_file = open('../Reviews/Yelp_Dataset/yelpzip/metadata', newline='', encoding="utf8")
    review_file = open('../Reviews/Yelp_Dataset/yelpzip/reviewContent', newline='', encoding="utf8")

    fake_outfile = open("../Reviews/Yelp_Dataset/fourAndFiveStarReviews/labeled_reviews_fake.txt", "w", encoding="utf8")
    true_outfile = open("../Reviews/Yelp_Dataset/fourAndFiveStarReviews/labeled_reviews_true.txt", "w", encoding="utf8")

    metadata_line = metadata_file.readline().split("\t")
    metadata_id = metadata_line[0]
    metadata_score = metadata_line[2]
    metadata_label = metadata_line[3]
    if metadata_label == "-1":
        metadata_label = "0"

    review_line = review_file.readline().split("\t")
    review_id = review_line[0]
    review_text = review_line[3]

    fake_amount = 0
    true_amount = 0
    total_amount = 0

    while metadata_id == review_id:
        if metadata_score == "4.0" or metadata_score == "5.0":
            if metadata_label == "0":
                fake_outfile.write(metadata_label + "\t" + review_text)
                fake_amount += 1
            elif metadata_label == "1":
                true_outfile.write(metadata_label + "\t" + review_text)
                true_amount += 1

            total_amount += 1

        try:
            metadata_line = metadata_file.readline().split("\t")
            metadata_id = metadata_line[0]
            metadata_score = metadata_line[2]
            metadata_label = metadata_line[3]
            if metadata_label == "-1":
                metadata_label = "0"

            review_line = review_file.readline().split("\t")
            review_id = review_line[0]
            review_text = review_line[3]

        except IndexError:
            break
            # End of file

    metadata_file.close()
    review_file.close()
    fake_outfile.close()
    true_outfile.close()

    print(fake_amount)
    print(true_amount)
    print(total_amount)


def get_labeled_review_reader():
    return open("../Reviews/Yelp_Dataset/allStarReviews/labeled_review.txt", "r", encoding="utf8")


def get_labeled_review_fake_reader():
    return open("../Reviews/Yelp_Dataset/allStarReviews/labeled_reviews_fake.txt", "r", encoding="utf8")


def get_labeled_review_true_reader():
    return open("../Reviews/Yelp_Dataset/allStarReviews/labeled_reviews_true.txt", "r", encoding="utf8")


def get_labeled_review_validation_true_reader():
    return open("../Reviews/Yelp_Dataset/machineLearningSets/labeled_reviews_validation_true.txt", "r", encoding="utf8")


def get_labeled_review_validation_fake_reader():
    return open("../Reviews/Yelp_Dataset/machineLearningSets/labeled_reviews_validation_fake.txt", "r", encoding="utf8")

def get_balanced_sample_reader(sample_id):
    return open("../Reviews/Yelp_Dataset/samples/sample_" + str(sample_id) + ".txt", "r", encoding="utf8")


def get_next_review_and_label(reader):
    next_line = reader.readline().split("\t")

    if next_line[0] == '':
        #print("End of file reached!")
        return "-1", "-1"

    label = next_line[0]
    # used https://regexr.com/ and
    # https://stackoverflow.com/questions/32698614/python-re-sub-only-replace-part-of-match?rq=1
    #
    # Replace all things divided by / that are not numbers by a space, up to 4
    # Normalize all unicode to show it in the string
    # Replace newline characters with nothing, as they serve no purpose for the context

    review = sub(r"([^0-9 /]+) ?/ ?([^0-9 /]+) ?/? ?([^0-9 /]+)? ?/? ?([^0-9 /]+)?", r"\1 \2 \3 \4",
                 normalize(u'NFC', next_line[1]).replace(r"\n", "").strip())

    return label, review


def find_class_distribution():
    counter = [0] * 2
    reader = get_labeled_review_reader()

    current_label, _ = get_next_review_and_label(reader)
    while True:
        if current_label == "-1":
            break
        elif current_label == "0":
            counter[0] += 1
        elif current_label == "1":
            counter[1] += 1

        current_label, _ = get_next_review_and_label(reader)

    total = counter[0] + counter[1]
    return "Total: " + str(total) + " Fake: amount:" + str(counter[0]) + " percentage:" + str((counter[0]/total) * 100) + \
           "%. Real: amount:" + str(counter[1]) + " percentage:" + str((counter[1]/total) * 100) + "%"


def find_longest_review_length():
    reader = get_labeled_review_reader()
    max_length = 0

    current_label, review = get_next_review_and_label(reader)
    while True:
        review_length = len(word_tokenize(review))
        if review_length > max_length:
            max_length = review_length
            print(max_length)

        if current_label == "-1":
            break

        current_label, review = get_next_review_and_label(reader)

    return max_length


def create_test_set():
    reader = get_labeled_review_reader()
    test_set_out = open("../Reviews/Yelp_Dataset/machineLearningSets/test_set.txt", "w", encoding="utf8")

    total_reviews = 608598
    test_set_20_percent = round(total_reviews * 0.2)

    for i in range(test_set_20_percent):
        label, review = get_next_review_and_label(reader)
        test_set_out.write(label + "\t" + review + "\n")

    reader.close()
    test_set_out.close()


def create_balanced_samples(sample_amount, upper_bound):
    for i in range(sample_amount):
        true_reader = get_labeled_review_validation_true_reader()
        fake_reader = get_labeled_review_validation_fake_reader()

        sample_outfile = open("../Reviews/Yelp_Dataset/samples/sample_" + str(i) + ".txt", "w", encoding="utf8")

        true_dist, fake_dist = us.perform_undersampling(upper_bound)

        amount_true = 0
        amount_fake = 0
        amount_total = 0

        # Write the sampled true reviews to the file
        counter = 0
        label, review = get_next_review_and_label(true_reader)

        while label != "-1":
            if len(true_dist) == 0:
                break
            if counter == true_dist[0]:
                true_dist.pop(0)

                sample_outfile.write(label + "\t" + review + "\n")
                amount_true += 1
                amount_total += 1

                label, review = get_next_review_and_label(true_reader)
                counter += 1
            else:
                label, review = get_next_review_and_label(true_reader)
                counter += 1

        # Write all the fake reviews to the file
        counter = 0
        label, review = get_next_review_and_label(fake_reader)

        while label != "-1":
            if len(fake_dist) == 0:
                break
            if counter == fake_dist[0]:
                fake_dist.pop(0)

                sample_outfile.write(label + "\t" + review + "\n")
                amount_fake += 1
                amount_total += 1

                label, review = get_next_review_and_label(fake_reader)
                counter += 1
            else:
                label, review = get_next_review_and_label(fake_reader)
                counter += 1

        print("Sample " + str(i) + " created.")
        print("True: " + str(amount_true))
        print("Fake: " + str(amount_fake))
        print("Total: " + str(amount_total))