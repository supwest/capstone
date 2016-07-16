import numpy as np
import pandas as pd
import graphlab as gl
import os
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
        song_data = gl.load_sf(path_to_song_data)
    else:
        song_data = gl.SFrame.read_csv(url)
    song_data = song_data[['song_id', 'title', 'artist_name']]
    song_data.rename({'song_id':'songid'})
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
    '''
    need to figure this out
    '''
    movie_users = movie_recommender.coefficients['user']['user']
    movie_users.add_row_number('id')
    user = movie_users[movie_users['id']==user]
    user_loadings = get_recommendations(movie_rec_matrix, user)
    print user_loadings.head()
    user_movies = movie_train[movie_train['user'] == user]
    print "user movies"
    print user_movies.head()
    user_movies = user_movies.sort('rating', ascending = False)
    print user_movies.head()
    # adding a 'rank' column, but just to join, it's not really a proper ranking of the movies
    user_movies = user_movies.add_row_number('rank') 
    print user_movies.head()
    movies_and_songs = song_list.join(user_movies, on='rank', how='inner')
    print movies_and_songs.head()
    return movies_and_songs



if __name__ == '__main__':
    song_recommender, song_train, song_test = load_songs()
    song_U = song_recommender.coefficients['userid']['factors']
    song_V = song_recommender.coefficients['songid']['factors']
    
    movie_recommender, movie_train, movie_song = load_movies()
    movie_U = movie_recommender.coefficients['user']['factors']
    movie_V = movie_recommender.coefficients['movie']['factors']

    movie_rec_matrix = np.dot(movie_U, np.transpose(song_V))

    '''
    the movie database doesn't have the movieid numbers so I'll have to get the     '''

    song_data = get_song_data()
    user = 0 #index of user
    recommendations = get_recommendations(movie_rec_matrix, user)
    #recommendations = recommendations.add_row_number('rank') #this will be the column for sorting later
    song_list = get_song_list(song_recommender, recommendations)
    comparison = get_comparisons(user, movie_rec_matrix, movie_train, song_list, movie_recommender)
    print comparison.head()
