import numpy as np
import pandas as pd
import graphlab as gl
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import NMF

class RatingsMatrix(object):
    '''
    class to create a toy ratings matrix to test on

    '''
    
    def __init__(self, user_loadings, item_loadings):
        self.user_loadings = user_loadings
        self.item_loadings = item_loadings


def make_matrix(user_list, n=3):
    num_rows = len(user_list)*n
    user1 = user_list[0]
    user2 = user_list[1]
    user3 = user_list[2]
    mat = np.zeros([num_rows, 3])
    user1_stack = user1
    user2_stack = user2
    user3_stack = user3
    for user in xrange(n):
        user1_stack = np.vstack((user1_stack, user1))
        user2_stack = np.vstack((user2_stack, user2))
        user3_stack = np.vstack((user3_stack, user3))
    mat1 = np.vstack((user1_stack, user2_stack, user3_stack))
    return mat1   

def add_noise(mat):
    #for row in xrange(mat.shape[0]):
    mat2 = mat.copy()
    for row in mat2:
        for i in xrange(len(row)):
            x = row[i]
            #print "x is {}".format(x)
            if x == 6:
                x -= np.random.choice([0,1,2])
            else:
                x += np.random.choice([0,1])
            #print "x is now {}".format(x)
            row[i] = x
    return mat2
        
def make_unordered_matrix(mat, user_list):
    mat[2] = user_list[1]
    mat[9] = user_list[0]
    mat[19] = user_list[1]
    mat[6] = user_list[2]
    mat[12] = user_list[2]
    return mat

def switch_columns(mat):
    mat2 = mat.copy()
    mat2 = mat2.T
    temp = mat2[0].copy()
    mat2[0]=mat2[1]
    mat2[1] = temp
    return mat2.T


if __name__ == '__main__':
    #action_loading = [.8, .1, .1]
    #mystery_loading = [.1, .8, .1]
    #romance_loading = [.1, .1, .8]
    
    #movie_user_loadings = [action_loading, mystery_loading, romance_loading]
    #song_user_loadings = [mystery_loading, romance_loading, action_loading]
    #movie_loadings = [action_loading, mystery_loading, romance_loading]
    #item_loadings = movie_loadings
    #song_loadings = [mystery_loading, romance_loading, action_loading]
    #movies = RatingsMatrix(movie_user_loadings, item_loadings)
   
    #movie_ratings = np.dot(np.array(movie_user_loadings), np.transpose(np.array(movie_loadings)))
    #song_ratings = np.dot(np.array(song_user_loadings), np.transpose(np.array(song_loadings)))
    action_user = np.array([6,1,1])
    mystery_user = np.array([1,6,1])
    romance_user = np.array([1,1,6])
    user_list = [action_user, mystery_user, romance_user]
    movie_ratings = make_matrix(user_list, 6)
    #print movie_ratings

    num_lf = 4#number of latent factors
    nmf = NMF(n_components = num_lf)
    movie_H = nmf.fit_transform(movie_ratings)
    movie_ratings_noise = add_noise(movie_ratings)
    dist_metric = 'cosine' # metric to use in distance calcs
    movie_dist = pairwise_distances(movie_H, metric=dist_metric)
    print movie_ratings_noise
    pairs = zip(movie_ratings, movie_dist[0])

    model_noise = NMF(n_components = num_lf)
    noise_H = model_noise.fit_transform(movie_ratings_noise)
    noise_dist = pairwise_distances(noise_H, metric=dist_metric)
    noise_pairs = zip(movie_ratings_noise, noise_dist[0])


    movie_ratings_unordered = make_unordered_matrix(movie_ratings, user_list)
    print movie_ratings_unordered
    model_unordered = NMF(n_components = num_lf)
    unordered_H = model_unordered.fit_transform(movie_ratings_unordered)
    unordered_dist = pairwise_distances(unordered_H, metric=dist_metric)
    unordered_pairs = zip(movie_ratings_unordered, unordered_dist[0])

    movie_ratings_both = add_noise(movie_ratings_unordered)
    model_both = NMF(n_components = num_lf)
    both_H = model_both.fit_transform(movie_ratings_both)
    both_dist = pairwise_distances(both_H, metric=dist_metric)
    both_pairs = zip(movie_ratings_both, both_dist[0])


    movie_ratings_switched = switch_columns(movie_ratings_both)
    model_switched = NMF(n_components=num_lf)
    switched_H = model_switched.fit_transform(movie_ratings_switched)

    order = np.argsort(np.dot(both_H[0], model_switched.components_))[::-1]
    one = np.dot(both_H[0], model_switched.components_.T[order[0]])
    two = np.dot(both_H[0], model_switched.components_.T[order[1]])
    three = np.dot(both_H[0], model_switched.components_.T[order[2]])

    

