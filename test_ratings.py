import pandas as pd
import numpy as np
from numpy.linalg import svd
from sklearn.metrics import pairwise_distances

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

    '''
    song to movie comparison
    '''
    movie_song_similarity = pairwise_distances(songs_V.T[0].reshape(1,-1), movies_V.T, metric='cosine')
    movies[np.argsort(movie_song_similarity[0])[::-1]]

    #with open('data/movie_titles.pkl', 'wb') as f:
    #   pickle.dump(f)
    #with open('data/similarity.pkl', 'wb') as f:
    #    pickle.dump(f)
