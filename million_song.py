import numpy as np
import pandas as pd
import graphlab as gl

def get_df():
    while True:
        ans = raw_input('does song SFrame exist?')
        yes = ['y', 'yes']
        if ans.lower() in set(yes):
            return gl.read_sf('data/ratings.sf')
        else:
            sf = gl.SFrame.read_csv(train_file, header=False, delimiter='\t', verbose=False)
            sf.rename({'X1':'userid', 'X2':'songid', 'X3':'listens'})
            sf.to_csv('data/ratings.sf')
            return sf


if __name__ == '__main__':
    song_sf = get_sf()
    song_train, song_test = song_sf.random_split(.8, seed=25)
    song_recommender = gl.recommender.factorization_recommender.create(song_train, user_id = 'userid', item_id = 'songid', target='listens')
    song_U = song_recommender.coefficients['userid']['factors']
    song_V = song_recommender.coefficients['songid']['factors']

    movie_sf = gl.SFrame.read_csv('http://static.turi.com/datasets/movie_ratings/sample.small', delimiter='\t', column_type_hints={'rating':int})
    movie_train, movie_test = movie_sf.random_split(.8, seed=25)
    movie_recommender = gl.recommender.factorization_recommender.create(movie_train, user_id = 'user', item_id = 'movie', target = 'rating')
    movie_U = movie_recommender.coefficients['user']['factor']
    movie_V = movie_recommender.coefficients['movie']['factor']

    movie_rec_mat = np.dot(movie_U, np.transpose(song_V))

    '''
    the movie database doesn't have the movieid numbers so I'll have to get the     '''

    songs = gl.SFrame.read_csv('https://static.turi.com/datasets/millionsong/song_data.csv')
    songs = songs[['song_id', 'title', 'artist_name']]

    recs = movie_rec_mat[0]
    song_list = song_recommender.coefficients['songid']['songid']
    song_list = gl.SFrame(song_list).rename({"X1":"song_id"})
    song_joined = songs.join(song_list, on='song_id', how='inner')

    for i in xrange(10):
        print song_recs[song_recs['song_id']==song_list[i]['song_id']]

    for i in recs[:10]:
        print movie_recommender.coefficients['movie']['movie'][i]

    '''
    to make getting the top recommedations out maybe add an 'index column' to song_list, join song_list to recs on song_list.index_column = recs.songid, then sort by index_column
    '''

