import random as rd

# 100% total set: 608598 Fake: amount:80466 percentage:13.22153539775024%.
#                        Real: amount:528132 percentage:86.77846460224976%


# Take a sample from the validation set. Return #upper_bound of random true and fake reviews
def perform_undersampling(true_amount, fake_amount, upper_bound):
    true_dist = rd.sample(range(0, true_amount), round(upper_bound/2))
    true_dist.sort()

    fake_dist = rd.sample(range(0, fake_amount), round(upper_bound/2))
    fake_dist.sort()

    return true_dist, fake_dist
