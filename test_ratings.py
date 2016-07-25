import pandas as pd
import numpy as np
from numpy.linalg import svd
from sklearn.metrics import pairwise_distances
import cPickle as pickle

'''
Script to read in csv file and make ratings matrix
'''

def get_titles(name):
    return name.split('-')[0].replace('[','').strip()

def clean_vals(df):
    for i in xrange(df.shape[0]):
        for j in xrange(df.iloc[i].shape[0]):
            v = df.iloc[i,j]
            #print type(v)
            if type(v) is str:
                a = np.float(v.split()[0])
                df.iloc[i, j] = a
            else:
                a = np.float(v)
                df.iloc[i,j] = a
            #print type(df.iloc[i,j])
    return pd.DataFrame(df, dtype='float')

if __name__ == '__main__':
    ratings_df = pd.read_csv('data/ratings.csv').drop(['Timestamp', 'Name', 'Banking Info (for science reasons)'], axis=1).dropna(axis=0)
    print (ratings_df.head())
    titles = ratings_df.columns.map(get_titles)
    print titles
    ratings_df.columns = titles
    songs_df = ratings_df[ratings_df.columns[[True if 'songs' in x.lower() else False for x in ratings_df.columns]]]
    songs_df.columns = songs_df.columns.map(lambda x: ' '.join(x.split()[1:]))
    movies_df = ratings_df[ratings_df.columns[[True if 'movies' in x.lower() else False for x in ratings_df.columns]]]
    movies_df.columns = movies_df.columns.map(lambda x: ' '.join(x.split()[1:]))

    
    songs_df = clean_vals(songs_df)
    movies_df = clean_vals(movies_df)
    songs = songs_df.columns
    movies = movies_df.columns
    song_mat = np.array(songs_df, dtype='float')
    movie_mat = np.array(movies_df, dtype='float')
    movies_U, movies_s, movies_V = svd(movie_mat, full_matrices=False)
    songs_U, songs_s, songs_V = svd(song_mat, full_matrices=False)

    pred_songs = np.dot(movies_U, np.dot(np.diag(movies_s), songs_V))
    song_suggestions = songs[np.argsort(pred_songs)]
    song_actual = songs[np.argsort(song_mat)]

    pred_movies = np.dot(songs_U, np.dot(np.diag(songs_s), movies_V))
    movie_suggestions = movies[np.argsort(pred_movies)]
    movie_actual = movies[np.argsort(movie_mat)]

    with open('data/actual_movies.pkl', 'w') as f:
        pickle.dump(movies_df, f)
    with open('data/actual_songs.pkl', 'w') as f:
        pickle.dump(songs_df, f)

    '''
    song to movie comparison
    '''
    movie_song_similarity = pairwise_distances(songs_V.T, movies_V.T, metric='cosine')
    movies[np.argsort(movie_song_similarity[0])[::-1]]

    with open('data/movie_titles.pkl', 'w') as f:
        pickle.dump(movies, f)
    with open('data/similarity.pkl', 'w') as f:
        pickle.dump(movie_song_similarity, f)

    '''
    using one movie rating to update preferences and then get a song
    '''
    #first get average user lf loadings from songs_U
    avg_loadings = np.mean(songs_U.T, axis=1)

    #get a movie for user to rate, just use first one for now
    exemplar_movie_title = movies[0]
    exemplar_movie_loadings = movies_V.T[0]


    '''
    holdout test
    '''
    holdout_id = 12
    holdout_song_ratings = songs_df.loc[holdout_id]
    holdout_movie_ratings = movies_df.loc[holdout_id]
    #for now easiest to holdout last row
    holdout_movie_df = movies_df[:-1]
    holdout_movie_U, holdout_movie_s, holdout_movie_V = svd(holdout_movie_df, full_matrices=False)
    holdout_pred_movie_ratings = np.dot(songs_U[-1][:-1], np.dot(np.diag(holdout_movie_s), holdout_movie_V))
    holdout_pred_movies = movies[np.argsort(holdout_pred_movie_ratings)[::-1]]
    holdout_actual_movies = movies[np.argsort(holdout_movie_ratings)[::-1]]

    '''
    get song movie rating matrix
    '''
    movie_song_ratings = np.dot(movies_V.T, np.dot(np.diag(movies_s), songs_V))
