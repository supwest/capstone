import numpy as np
import pandas as pd
import scipy.stats as scs
from numpy.linalg import svd
from sklearn.decomposition import NMF
from itertools import permutations
from bokeh.plotting import figure, show, output_file
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import cPickle as pickle
from multiprocessing import Pool
from time import time
#from pathos.multiprocessing import ProcessingPool
#import dill

def f((self, shuffle)):
    new_matrix = self._make_test_matrix(shuffle)
    return self._get_test_score(new_matrix)

class User(object):
    '''
    User class to hold user preferences.
    Matrix class will use these to sample from to get ratings
    '''

    def __init__(self, loadings):
        self.loadings = loadings

    def get_rating(self, item_loadings):
        
        #remove randomness for testing
        #--------------------
        random_loadings = self._generate_loadings()
        return np.dot(np.transpose(random_loadings), item_loadings)
        '''
        random_loadings = self.loadings
        return np.dot(random_loadings, item_loadings)
        '''
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
        self.items = self._make_items()
        self.users = self._make_rows()
        self.ratings = self._make_ratings()
        self.U, self.s, self.V = svd(self.ratings)
        self.H, self.W = self._nmf(self.ratings)


    def _make_items(self):
        return  np.array(np.random.choice(self.item_types.keys(), self.n_items))
    
    def _make_rows(self):
        return np.array(np.random.choice(self.user_types.keys(), self.n_users))

    def _make_ratings(self):
        ratings = np.zeros((self.n_users, self.n_items))
        for idx, user in enumerate(self.users):
            u = User(self.user_types[user])
            for idx2, item in enumerate(self.items):
                ratings[idx, idx2] = u.get_rating(self.item_types[item])
        return ratings
            
    def __getitem__(self, idx):
        return self.ratings[idx]

    def _nmf(self, ratings):
        model = NMF()
        h = model.fit_transform(ratings)
        return h, model.components_


class Tester(object):
    '''
    Class to test matrix decompositions
    takes two RatingsMatrix objects as inputs

    '''
   
    def __init__(self, matrix_1, matrix_2, s_option='user', decomp_type='svd'):
        self.matrix_1 = matrix_1
        self.matrix_2 = matrix_2
        self.s_option = s_option
        self.decomp_type = decomp_type
        self.true_matrix = self._make_true_matrix()
        #self.test_matrix = self._make_test_matrix()

    def _make_true_matrix(self):
        '''
        Input: none
        Output: matrix

        makes the 'true' rating matrix from the matrix_1 rows(users) and the
        matrix_2 columns(items)
        '''

        '''
        test
        u = np.array([self.matrix_1.user_types[user] for user in self.matrix_1.users])
        i = np.array([self.matrix_2.item_types[item] for item in self.matrix_2.items])
        m = np.dot(u, i.T)
        
        '''
        users = self.matrix_1.users
        items = self.matrix_2.items
        m = np.zeros([len(users), len(items)])
        for idx, user in enumerate(users):
            for idx2, item in enumerate(items):
                m[idx, idx2] = np.dot(self.matrix_1.user_types[user], self.matrix_2.item_types[item])
        return m

    def _make_test_matrix(self, matrix, test_decomp='svd'):
        '''
        Input: a matrix
        Output: a recomposed estimated ratings matrix

        Decomposes input matrix according to decomposition type
        and then makes an estimated ratings matrix
        '''
        if test_decomp == 'svd':
            _, s1, V = svd(matrix)
            how = self.s_option
            how = self.test_how
            #print "s1", s1
            #print "how", how
            s = self._get_s(s1, how)
            #print s
            #print V
            #print self.matrix_1.U
            return np.dot(self.matrix_1.U, np.dot(s, V))
        elif test_decomp == 'nmf':
            model = NMF()
            H = model.fit_transform(matrix)
            print H
            W = model.components_
            return np.dot(self.matrix_1.H, W)
        else:
            pass

        '''
        if matrix is None:
            if self.decomp_type == 'svd':
                item_matrix = self.matrix_2.V
                s = self._get_s()
                return np.dot(self.matrix_1.U, np.dot(s, item_matrix))
        else:
            if self.decomp_type == 'svd':
                _, s, item_matrix = svd(matrix)
                return np.dot(self.matrix_1.U, np.dot(self.get_s
            else:
                model = NMF()
                item_matrix = model.fit_transform(matrix).components_
        if self.decomp_type == 'svd': 
                    elif self.decomp_type == 'nmf':
            return np.dot(self.matrix_1.H, item_matrix)
        '''
        
         

    def _get_s(self, s, how):
        '''
        Input: matrix s, and string how
        Output: square matrix s for use in reconstructing ratings matrix

        returns the s matrix to use when combining matrices from svd decomposition:
        if how is 'user' then use the s from the user matrix
        if how is 'item' then use the s from the item matrix
        if how is 'combined' then use the sqrt of the product of both s matrices
        '''
        how = self.test_how
        if how == 'user':
            return np.diag(self.matrix_1.s)
        elif how == 'item':
            return np.diag(s)
        elif how == 'combined':
            return np.sqrt(np.dot(np.diag(self.matrix_1.s), np.diag(s)))
        else:
            return np.eye(np.diag(self.matrix_1.s).shape)

    def _get_test_score(self, matrix, test_decomp='svd'):
        '''
        Input: matrix
        Output: float

        caclulates MSE between the true matrix and the test matrix
        '''
        test_matrix = self._make_test_matrix(matrix, test_decomp)
        score = np.mean((test_matrix - self.true_matrix)**2)
        return score

    def test(self, how='user', test_decomp='svd'):
        '''
        Input: None
        Output: list of tuples (shuffle, score)
        shuffle is the order of the columns 
        score is the MSE between the test and true matricies
        '''
        self.test_how = how
        idxes = [i for i in xrange(len(self.matrix_2.items))]
        shuffles = permutations(idxes)
        scores = []
        for shuffle in shuffles:
            #print shuffle
            new_matrix = self._make_shuffled_matrix(shuffle)
            #print new_matrix
            score = self._get_test_score(new_matrix, test_decomp)
            scores.append((shuffle, score))
        return scores

    def parallel_test(self, how='user'):

        self.test_how = how
        idxes = [i for i in xrange(len(self.matrix_2.items))]
        shuffles = permutations(idxes)
        print shuffles
        #shuffs = [x for x in shuffles]
        shuffles2 = np.array([x for x in shuffles])
        print shuffles2
        p = Pool(2)
        #p = ProcessingPool(nodes=2)
        result = p.map(f, [self]*len(shuffles2), shuffles)
        p.close()
        p.join()
        return result

    def _f(self, shuffle):
        new_matrix = self._make_test_matrix(shuffle)
        return self._get_test_score(new_matrix)

    def _make_shuffled_matrix(self, shuffle):
        '''
        Input: List
        output matrix

        reorders the columns of a matrix according to the order in shuffle
        '''
        temp = np.zeros_like(self.matrix_2.ratings)
        for idx,i in enumerate(shuffle):
            temp[idx] = np.transpose(self.matrix_2.ratings)[i]
        #print temp
        temp2 = np.transpose(temp)
        #print temp
        return temp2

def pretty_plot(m):
    p = figure(title='Error')
    p.xaxis.axis_label = "Items"
    p.yaxis.axis_label = "Users"
    cmap=plt.cm.get_cmap("inferno")

    cmap = plt.cm.YlOrRd_r
    x = [i for i in xrange(m.shape[1])]
    y = [i for i in xrange(m.shape[0])]
    #print x
    #print y
    for i in x:
        for j in y:
            plt.scatter(x = x[i],y = y[j], s=1000, marker='s', color=cmap(m[i,j]))
    #output_file('a.html')
    #show(p)
    plt.show()
   # return p



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


    #test_loadings = [.3, .6, .1, .9]
    #i_loadings = [.1, .8, .3, .1]
    #test = User(test_loadings)
    #test_rating = test.get_rating(i_loadings)
    #print test_rating

    #test_matrix = RatingsMatrix(4, 4, item_loadings, user_loadings)
    #print test_matrix.columns
    #print test_matrix.columns.shape
    #print test_matrix.users
    #print test_matrix.columns[0]

    #print test_matrix.ratings

    num_users = 8
    num_items = 8
    movies = RatingsMatrix(num_users, num_items, item_loadings, user_loadings)
    songs = RatingsMatrix(num_users, num_items, item_loadings, user_loadings)

    test = Tester(movies, songs)
    #print test.true_matrix
    #out = test.test('item')
    #scores_only = [s[1] for s in out]
    #with open('item_s.pkl', 'wb') as f:
    #    pickle.dump(out, f)
    #plt.hist(scores_only)
    #plt.show()
    #plt.close()


    start = time()
    svd_user = test.test('user')
    print time()-start

    start = time()
    svd_item = test.test('item')
    print time()-start
    
    start=time()
    svd_square = test.test('combined')
    print time()-start

    user_scores = np.array([s[1] for s in svd_user])
    user_orders = np.array([s[0] for s in svd_user])

    item_scores = np.array([s[1] for s in svd_item])
    item_orders = np.array([s[0] for s in svd_item])

    square_scores = np.array([s[1] for s in svd_square])
    square_orders = np.array([s[0] for s in svd_square])

    plt.hist(user_scores, alpha=.4, color='g', normed=1, label='user sigma')
    plt.hist(item_scores, alpha=.4, color='b', normed=1, label='item sigma')
    plt.hist(square_scores, alpha=.4, color='r', normed=1, label='combined sigma')
    plt.legend(loc='best')
    plt.savefig('{0}by{1}_4.png'.format(num_users, num_items))
    plt.close()
    
    user_sorted = user_orders[np.argsort(user_scores)[::-1]]
    item_sorted = item_orders[np.argsort(item_scores)[::-1]]
    square_sorted = square_orders[np.argsort(square_scores)[::-1]]

    print zip(user_sorted, item_sorted, square_sorted)[:5]

    #start=time()
    #a = test.parallel_test('user')
    #print time()-start
   
    start = time()
    nmf_test = test.test(test_decomp='nmf')
    print time()-start
