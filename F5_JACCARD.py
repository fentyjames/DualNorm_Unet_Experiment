from __future__ import print_function


def Jaccard(y, y_pred, epsilon=1e-8):   
    TP = (y_pred * y).sum(0)
    FP = ((1-y_pred)*y).sum(0)
    FN = ((1-y)*y_pred).sum(0)
    jack = (TP+epsilon) / (TP+FP+FN+epsilon)
    return jack

# #%%
# import random

# #random_number = random.randint(0, 55)
# random_range = random.sample(range(56), k=55)
# #%%
# print(random_range,"\n")
# #print(random_number)
# # %%
