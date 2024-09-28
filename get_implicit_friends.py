import collections
from tqdm import tqdm
import utils

# Return a collection of items with user positive sentiment
def getSet_positive(triples):
    user_items = collections.defaultdict(set)
    for u,i,r in triples:
        if r == 1:
            user_items[u].add(i)
    return user_items

# Return a collection of items with user negative sentiment
def getSet_negative(triples):
    user_items = collections.defaultdict(set)
    for u,i,r in triples:
        if r == 0:
            user_items[u].add(i)
    return user_items
# Using the K-Nearest Neighbors (KNN) algorithm to obtain the K nearest neighbors of a user.
def knn(trainset,k,sim_method):
    sims = {}
    for e1 in tqdm(trainset):
        ulist = []
        for e2 in trainset:
            if e1 == e2 or len(trainset[e1] & trainset[e2]) == 0:
                continue
            sim = sim_method(trainset[e1], trainset[e2])
            ulist.append((e2, sim))
        ulist = sorted(ulist, key=lambda x: x[1], reverse=True)
        sims[e1] = [i for i in ulist[:k]]
    return sims  # {u1:[]]

"""Finding positive sentiment implicit friends"""
def get_positive_friends(train_data):
    pairs = []
    weights = []
    user_items = getSet_positive( train_data )
    user_sims = knn(user_items,5, cos4set)
    # user_sims = knn(user_items,1, cos4set)
    for key, value in user_sims.items():
        if len(value) != 0:
            for user,sims in value:
                pairs.append((int(key),int(user)))
                weights.append(sims)
    return pairs,weights
"""Finding negative sentiment implicit friends"""
def get_negative_friends(train_data):
    pairs = []
    weights = []
    user_items = getSet_negative( train_data )
    user_sims = knn(user_items,5, cos4set)
    # user_sims = knn(user_items,1, cos4set)
    for key, value in user_sims.items():
        if len(value) != 0:
            for user,sims in value:
                pairs.append((int(key),int(user)))
                weights.append(sims)
    return pairs,weights

def cos4set( set1, set2 ):
    return len(set1&set2)/(len(set1)*len(set2))**0.5

if __name__ == '__main__':
    pass





