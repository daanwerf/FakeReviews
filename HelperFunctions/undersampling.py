import random as rd

# 100% total set: 608598 Fake: amount:80466 percentage:13.22153539775024%.
#                        Real: amount:528132 percentage:86.77846460224976%

# 20% Test set: 121720 Fake: amount:16291 percentage:13.383996056523168%.
#                      Real: amount:105429 percentage:86.61600394347683%

# 80% Training validation: 486878 Fake: amount:64175 percentage:13.180920066217821%.
#                                 Real: amount:422703 percentage:86.81907993378218%


# Take a sample from the validation set. Return #upper_bound of random true and fake reviews
def perform_undersampling(upper_bound):
    true_dist = rd.sample(range(0, 422703), round(upper_bound/2))
    true_dist.sort()

    fake_dist = rd.sample(range(0, 64175), round(upper_bound/2))
    fake_dist.sort()

    return true_dist, fake_dist
