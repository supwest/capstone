import numpy as np
import pandas as pd
import graphlab as gl








if __name__ == '__main__':
    song_recommender = gl.load_model('song_recommender')
    song_coeffs = song_recommender.coefficients['songid']
    song_cluster_input = {sid:song_coeffs[['songid']==sid]['factors'] for sid in song_coeffs['songid']}
    song_cluster_sframe = gl.SFrame(song_cluster_input)
    #train, test = song_cluter_sframe.random_split(.9, seed=25)
    song_cluster_sframe = gl.SFrame(song_coeffs['factors']).unpack('X1', column_name_prefix='factor')
    song_cluster = gl.nearest_neighbors.create(song_cluster_sframe)
    song_data = gl.load_sframe('data/song_data')
    target_song_id = song_data[song_data['artist_name']=='The Wreckers'][4]['songid']
    target_song_factors = song_coeffs[song_coeffs['songid']==target_song_id]['factors']
    target_song_factors = gl.SFrame(target_song_factors).unpack('X1', column_name_prefix='factor')

    top5index = song_cluster.query(target_song_factors, k=10)['reference_label']
    song_coeffs = song_coeffs.add_row_number()
    top5songs = []
    for index in top5index:
        print index
        new_id = song_coeffs[song_coeffs['id']==index]['songid'][0]
        print new_id
        new_song = song_data[song_data['songid'] == new_id]
        top5songs.append(new_song)
    #new_id = song_coeffs.apply(lambda x: song_coeffs[song_coeffs['id'] ==x] for x in top5index)
    print top5songs


