import numpy as np
import pandas as pd
import graphlab as gl
import os
from sklearn.metrics import pairwise_distances
def get_sf():
    if os.exists['song_recommender']:
        song_sf = gl.load_model('song_recommender')
        if ans.lower() in set(yes):
            return gl.read_sf('data/ratings.sf')
        else:
            sf = gl.SFrame.read_csv(train_file, header=False, delimiter='\t', verbose=False)
            sf.rename({'X1':'userid', 'X2':'songid', 'X3':'listens'})
            sf.to_csv('data/ratings.sf')
            return sf

def load_songs():
    path_to_sf = 'data/song_sf'
    url = 'https://static.turi.com/datasets/millionsong/10000.txt'
    if os.path.exists(path_to_sf):
        song_sf = gl.load_sframe(path_to_sf)
    else:
        song_sf = gl.SFrame.read_csv(url, header=False, delimiter='\t')
        song_sf.rename({"X1":'userid', 'X2':'songid', 'X3':'listens'})
        song_sf.save('data/song_sf')
    train, test = song_sf.random_split(.8, seed=25)
    if os.path.exists('song_recommender'):
        song_recommender = gl.load_model('song_recommender')
    else:
        song_recommender = gl.recommender.factorization_recommender.create(train, user_id='userid', item_id='songid', target='listens')
    return song_recommender, train, test


def load_movies():
    path_to_movie_sf = 'data/movie_sf'
    url = 'https://static.turi.com/datasets/movie_ratings/sample.small'
    if os.path.exists(path_to_movie_sf):
        movie_sf = gl.load_sframe(path_to_movie_sf)
    else:
        movie_sf = gl.SFrame.read_csv(url, delimiter='\t', column_type_hints={'rating':int})
    train, test = movie_sf.random_split(.8, seed=25)
    if os.path.exists('movie_recommender'):
        movie_recommender = gl.load_model('movie_recommender')
    else:
        movie_recommender = gl.recommender.factorization_recommender.create(train, user_id='user', item_id='movie', target='rating')
    movie_sf.save('data/movie_sf')
    return movie_recommender, train, test

def get_song_data():
    path_to_song_data = 'data/song_data'
    url = 'https://static.turi.com/datasets/millionsong/song_data.csv'
    if os.path.exists(path_to_song_data):
        song_data = gl.load_sframe(path_to_song_data)
        
    else:
        song_data = gl.SFrame.read_csv(url)
        song_data = song_data[['song_id', 'title', 'artist_name']]
        song_data.rename({'song_id':'songid'})
    #song_data = song_data[['song_id', 'title', 'artist_name']]
    
    song_data.save('data/song_data')
    return song_data

def get_recommendations(m, user):
    '''
    gets predicted ratings for user, sorts them, adds rank column to sort by later.
    Returns an SFrame of indcies to and rank
    '''
    recs = gl.SFrame(np.argsort(m[user])).rename({'X1':'index'})
    recs = recs.add_row_number('rank')
    return recs

def get_song_list(song_recommender, recommendations):
    song_list = song_recommender.coefficients['songid']['songid']
    song_list = gl.SFrame(song_list).rename({"X1":"songid"})
    song_list = song_list.add_row_number('index') #add this column to join song_list to recommendeations
    song_list = song_list.join(recommendations, on='index', how='inner')
    
    song_list = song_list.join(song_data, on='songid', how='inner')
    song_list = song_list.sort('rank', ascending=True) 
    song_list.save('song_recommendations')
    return song_list

def get_comparisons(user, movie_rec_matrix, movie_train, song_list, movie_recommender):
    movie_users = movie_recommender.coefficients['user']['user']
    movie_users = gl.SFrame(movie_users).rename({"X1":'user'}).add_row_number('id')
    user_name = movie_users[movie_users['id']==user]['user'][0]
    user_loadings = get_recommendations(movie_rec_matrix, user)
    user_movies = movie_train[movie_train['user'] == user_name]
    user_movies = user_movies.sort('rating', ascending = False)
    user_movies = user_movies.add_row_number('rank') 
    movies_and_songs = song_list.join(user_movies, on='rank', how='inner')
    movies_and_songs = movies_and_songs['user', 'title', 'artist_name', 'movie']
    return movies_and_songs

def get_factors(recommender, name):
    '''
    takes recommender model, and returns right matrix of decompostion:
    Matrix with latent factors as rows and items (i.e. movies or songs) as columns
    '''
    factors = recommender.coefficients[name]['factors']
    latent_factors = []
    for i in xrange(len(factors[0])):
        lf = [x[i] for x in factors]
        latent_factors.append(lf)
    return latent_factors

def reorder_factors_cosine(lf1, lf2):
    '''
    takes arrays of latent factors and reorders lf2 by correlation with latent factors in lf1.
    
    This reordering is just pairwise using cosine sim, but the correlation depends on column order which is arbitrary.

    So I'll try other measures to order latent factors
    '''

    min_dim = min(lf1.shape[1], lf2.shape[1])
    new_lf2 = np.array([lf[:min_dim] for lf in lf2])
    new_lf = []
    for idx, lf in enumerate(lf1):
        #print idx
        #print lf
        order = np.argsort(pairwise_distances(lf[:min_dim].reshape(1,-1), new_lf2))
        #print new_lf2
        #print order[0]
        new_lf2 = new_lf2[order[0]]
        #print new_lf2
        new_lf.append(new_lf2[0])
        new_lf2 = new_lf2[1:]
        #print new_lf2
    #print new_lf
    return new_lf
            

def reorder_factors_variance(lf1, lf2):
    var1 = np.var(lf1, axis=1)
    lf1_ordered = lf1[np.argsort(var1)]
    var2 = np.var(lf2, axis=1)
    lf2_ordered = lf2[np.argsort(var2)]
    #lf2_reordered = lf2[np.argsort(var2)[np.argsort(var1)]]
    #lf2_reordered = lf2[np.argsort(var1)[np.argsort(var2)]]
    return lf1_ordered, lf2_ordered


if __name__ == '__main__':
    song_recommender, song_train, song_test = load_songs()
    song_U = song_recommender.coefficients['userid']['factors']
    song_V = song_recommender.coefficients['songid']['factors']
    
    movie_recommender, movie_train, movie_song = load_movies()
    movie_U = movie_recommender.coefficients['user']['factors']
    movie_V = movie_recommender.coefficients['movie']['factors']

    #movie_rec_matrix = np.dot(movie_U, np.transpose(song_V))

    '''
    the movie database doesn't have the movieid numbers so I'll have to get the     '''

    song_data = get_song_data()
    #user = 866 #index of user
    #user = 2956
    #user = 9379
    user = np.random.choice(len(movie_U))
    #user = 2956
    print len(movie_U)
    print user

    '''
    before generating recommendations, reorder right song matrix

    to use cosine similarity: from sklearn.metrics import pairwise_distance
    from scipy.spatial.distance import cosine
    
    '''

    movie_factors = np.array(get_factors(movie_recommender, 'movie'))
    song_factors = np.array(get_factors(song_recommender, 'songid'))
    #print movie_factors
    #print song_factors
    #a = raw_input("?")
    nlf = reorder_factors_cosine(movie_factors, song_factors)
    nlf1, nlf2 = reorder_factors_variance(movie_factors, song_factors)
    #print nlf
    #print nlf2
    #movie_rec_matrix = np.dot(movie_U, nlf2)
    #print nlf1.shape
    #print nlf2.shape
    #print movie_U.shape
    #raw_input("?")

    movie_rec_matrix = np.dot(movie_U, nlf2)

    recommendations = get_recommendations(movie_rec_matrix, user)
    song_list = get_song_list(song_recommender, recommendations)
    comparison = get_comparisons(user, movie_rec_matrix, movie_train, song_list, movie_recommender)
    print comparison.head()
