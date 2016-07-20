import numpy as np
import pandas as pd
import scipy.stats as scs
from numpy.linalg import svd
from sklearn.decomposition import NMF

class User(object):
    '''
    User class to hold user preferences.
    Matrix class will use these to sample from to get ratings
    '''

    def __init__(self, loadings):
        self.loadings = loadings

    def get_rating(self, item_loadings):
        random_loadings = self._generate_loadings()
        #print random_loadings
        return np.dot(np.transpose(random_loadings), item_loadings)

    def _generate_loadings(self):
        return np.array([scs.uniform(l-.1, l+.1).rvs(1) for l in self.loadings])



class RatingsMatrix(object):
    '''
    class to build a ratings matrix
    '''

    def __init__(self, n_users, n_items, item_types, user_types):
        self.n_users = n_users
        self.n_items = n_items
        self.item_types = item_types
        self.user_types = user_types
        self.columns = self._make_columns()
        self.users = self._make_rows()
        self.ratings = self._make_ratings()
        self.U, self.s, self.V = svd(self.ratings)


    def _make_columns(self):
        return  np.array(np.random.choice(self.item_types.keys(), self.n_items))
    
    def _make_rows(self):
        return np.array(np.random.choice(self.user_types.keys(), self.n_users))

    def _make_ratings(self):
        ratings =np.zeros((self.n_users, self.n_items))
        for idx, user in enumerate(self.users):
            u = User(self.user_types[user])
            for idx2, item in enumerate(self.columns):
                ratings[idx, idx2] = u.get_rating(self.item_types[item])
        return ratings
            
    def __getitem__(self, idx):
        return self.ratings[idx]

if __name__ == '__main__':
    '''
    action movies and songs: [.2, .9, .4, .1]
    mystery movies and songs: [.1, .3, .9, .1]
    romance movies and songs: [.9, .2, .1, .7]
    horror movies and songs: [.4, .2, .5, .8]

    blue users: [.3, .1, .8, .2]
    green users: [.7, .2, .1, .3]
    red users: [.4, .9, .4, .1]
    orange users: [.2, .4, .3, .9]

    '''

    user_types = ['blue', 'green', 'red', 'orange']
    item_types = ['action', 'mystery', 'romance', 'horror']
    user_loadings = {'blue':[.3, .1, .8, .2], 'green': [.7, .2, .1, .3]
, 'red': [.4, .9, .4, .1], 'orange': [.2, .4, .3, .9]}
    item_loadings = {'action': [.2, .9, .4, .1], 'mystery': [.1, .3, .9, .1], 'romance': [.9, .2, .1, .7], 'horror': [.4, .2, .5, .8]}


    test_loadings = [.3, .6, .1, .9]
    i_loadings = [.1, .8, .3, .1]
    test = User(test_loadings)
    test_rating = test.get_rating(i_loadings)
    print test_rating

    test_matrix = RatingsMatrix(4, 4, item_loadings, user_loadings)
    print test_matrix.columns
    print test_matrix.columns.shape
    print test_matrix.users
    print test_matrix.columns[0]

    print test_matrix.ratings

    songs = RatingsMatrix(4,4,item_loadings, user_loadings)
    movies = RatingsMatrix(4,4, item_loadings, user_loadings)
