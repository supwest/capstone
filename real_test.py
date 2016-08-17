
'''
script to test the movies and song matrices from the google forms survey
'''
import pandas as pd
import numpy as np
from numpy.linalg import svd
import cPickle as pickle
#from fancyimpute import NuclearNormMinimization, KNN, BiScaler, SoftImpute, SimpleFill, IterativeSVD
import graphlab as gl
import json
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

def matrix_concat(m1, m2, axis=0):
    '''
    takes two matrix objects and returns concatenated Matrix object
    '''
    return Matrix(pd.concat([m1.matrix,m2.matrix], axis=axis))

class Matrix(object):
    '''
    Matrix class to hold the ratings matrix and decomposition matrices
    '''

    def __init__(self, matrix):
        '''
        takes a pandas df with users as rows, items as cols and ratings as values
        '''
        self.matrix = matrix
        self.U, self.s, self.V = svd(self.matrix, full_matrices=False)
        self.items = matrix.columns
        self.users = matrix.index

    def split(self, idx=4):
        '''
        Input: Int
        output df, df
        splits the matrix into two seprate dfs:
        self.matrix[:idx]
        self.matrix[idx:]
        Returns them, and they
        can be input for new Matrix objects
        '''
        df1 = self.matrix[:idx].copy().reset_index()
        df2 = self.matrix[idx:].copy().reset_index()
        return Matrix(df1), Matrix(df2)

    
class PredictedMatrix(object):
    '''
    class to make a prediction matrix
    '''
    def __init__(self, user_source, item_source):
        '''
        Input: Matrix object, Matrix object
        '''
        self.user_source = user_source
        self.item_source = item_source
        self.U = self.user_source.U
        self.s = self.user_source.s
        self.V = self.item_source.V
        self.items = self.item_source.items
        self.users = self.user_source.users
        self.matrix = self._fit()

    def _fit(self):
        m = np.dot(self.U, np.dot(np.diag(self.s), self.V))
        return pd.DataFrame(m, index = self.users, columns=self.items)

    def split(self, idx=4):
        df1 = self.matrix[:idx].copy().reset_index()
        df2 = self.matrix[:idx].copy().reset_index()
        return Matrix(df1), Matrix(df2)

class Tester(object):
    '''
    Tester object to run tests on ratings matrices
    '''
    
    def __init__(self, m1, m2):
        '''
        Input: m1, m2 both instances of the Matrix class
        '''
        self.matrix_1 = m1
        self.matrix_2 = m2
        self.error_matrix = m1.matrix-m2.matrix
        self.mse = self._calc_mse(self.matrix_1.matrix, self.matrix_2.matrix)
        #self.rank_corr = self._calc_rank_corr()

    def _calc_mse(self, matrix1, matrix2):
        '''
        Input: df
        Output: df

        takes a 'true matrix' and returns the MSE of the difference
        between the true matrix and the fitted matrix

        '''
        ea=matrix1 - matrix2
        #ea = matrix1.reset_index(drop=True) -matrix2.reset_index(drop=True)
        #print ea
        #sq_e = e**2
        #sum_sq_e = np.mean(np.sum(sq_e))
        #self.error_ = sum_sq_e

        #return np.mean(np.square(ea)).mean()
        return np.mean(np.square(self.error_matrix)).mean()

    def _calc_rank_corr(self):
        return spearmanr(self.matrix_1.matrix, self.matrix_2.matrix, axis=1)


    def common_user_test(self, n_common=1):
        '''
        
        '''
        m1, m2 = self.matrix_1.split()
        m1_temp = Matrix(m1.matrix.copy())
        s1, s2 = self.matrix_2.split()
        pred_mat = PredictedMatrix(m1_temp,s2)
        t1 =  Tester(s2, pred_mat)
        mse_1 = t1.mse
        for idx in xrange(n_common):
            m1_temp.matrix.iloc[idx] = m2.matrix.iloc[idx]
        new_m1 = Matrix(m1_temp.matrix)
        pred_mat2 = PredictedMatrix(new_m1, s2)
        t2 = Tester(s2, pred_mat2)
        mse_2 = t2.mse
        return mse_2-mse_1

    def block_matrix_test(self):
        '''
        Generates a block diagonal matrix to test:
        [[m1, s2_0], [m1_0, s2]]

        '''
        idx = self.matrix_1.shape[0]/2
        m1 = self.matrix_1.matrix[:idx].copy()
        m2 = self.matrix_1.matrix[idx:].copy()
        s1 = self.matrix_2.matrix[:idx].copy()
        s2 = self.matrix_2.matrix[idx:].copy()
        m1_zeros = pd.DataFrame(0, index=s2.index, columns=m1.columns)
        s2_zeros = pd.DataFrame(0, index=m1.index, columns=s2.columns)
        t1 = pd.concat([m1, s2_zeros], axis=1)
        t2 = pd.concat([m1_zeros, s2], axis=1)
        temp = pd.concat([t1, t2])
        full = pd.concat([self.matrix_1.matrix, self.matrix_2.matrix], axis=1)
        temp = Matrix(temp)
        full = Matrix(full)
        temp_pred = Matrix(pd.DataFrame(np.dot(temp.U, np.dot(np.diag(temp.s), temp.V)), index=full.users, columns=full.items))
        m1_test = Matrix(m1)
        m2_test = Matrix(m2)
        s1_test = Matrix(s1)
        s2_test = Matrix(s2)
        m2_pred = PredictedMatrix(m2_test, s1_test)



def block_diagonal(m, s):
    '''
    return block diagonal matrix, and full matrix as Matrix objects
    '''

    idx = m.matrix.shape[0]/2
    #print idx
    m1 = m.matrix[:idx].copy()
    m2 = m.matrix[idx:].copy()
    s1 = s.matrix[:idx].copy()
    s2 = s.matrix[idx:].copy()
    m1_zeros = pd.DataFrame(0, index = s2.index, columns=m1.columns)
    s2_zeros = pd.DataFrame(0, index = m1.index, columns=s2.columns)
    m1_nan = pd.DataFrame(np.nan, index=s2.index, columns=m1.columns)
    s2_nan = pd.DataFrame(np.nan, index=m1.index, columns=s2.columns)
    t1 = pd.concat([m1, s2_zeros], axis=1)
    t2 = pd.concat([m1_zeros, s2], axis=1)
    temp_zeros = pd.concat([t1, t2])
    #print "temp"
    #print temp
    f1 = pd.concat([m1, s1], axis=1)
    f2 = pd.concat([m2, s2], axis=1)
    full = pd.concat([f1,f2])
    t1_nan = pd.concat([m1, s2_nan], axis=1)
    t2_nan = pd.concat([m1_nan, s2], axis=1)
    temp_nan = pd.concat([t1_nan, t2_nan])
    return temp_nan, Matrix(temp_zeros), Matrix(full)

def subset_matrix(mat, idx=None, how='top'):
    '''
    Takes Matrix object and returns a Matrix object made from a subset
    of the input Matrix.
    idx determines the size of the subset. If idx is not specified 
    idx = mat.matrix.shape[0]/2
    If how == 'top', subset_matrix returns mat.matrix[:idx]
    else return mat.matrix[idx:]
    '''
    if idx is None:
        idx = mat.matrix.shape[0]/2
    if how == 'top':
        return Matrix(mat.matrix[:idx])
    else:
        return Matrix(mat.matrix[idx:])

def build_full_predicted_matrix(m1, m2):
    '''
    takes 2 Matrix objects returns a matrix of actual obsevations and
    predictions
    '''
    pred_1 = PredictedMatrix(m1, m2)
    pred_2 = PredictedMatrix(m2, m1)
    pred_top = matrix_concat(m1, Matrix(pred_1.matrix), 1)
    pred_bottom = matrix_concat(Matrix(pred_2.matrix), m2, 1)
    return matrix_concat(pred_top, pred_bottom)


def build_interleaved_matrix(m, s, fill='zero'):
    cols = [c for c in np.concatenate([m.matrix.columns, s.matrix.columns]) if c != 'id']
    
    a_locs = m.matrix.index[xrange(0, m.matrix.shape[0], 2)]
    b_locs = m.matrix.index[xrange(1, m.matrix.shape[0], 2)]
    if fill == 'zero':
        insert = np.hstack([m.matrix.loc[a_locs], np.zeros(s.matrix.loc[a_locs].shape)])
        insert2 = np.hstack([np.zeros(m.matrix.loc[b_locs].shape), s.matrix.loc[b_locs]])
    else:
        insert = np.hstack([m.matrix.loc[a_locs], np.zeros(s.matrix.loc[a_locs].shape)*np.nan])
        insert2 = np.hstack([np.zeros(m.matrix.loc[b_locs].shape)*np.nan, s.matrix.loc[b_locs]])

    df1 = pd.DataFrame(insert, index=a_locs, columns=cols)
    df2 = pd.DataFrame(insert2, index=b_locs, columns=cols)
    return Matrix(pd.concat([df1, df2]).sort_index())   

def get_melted(m):
    ids = [i for i in m.index]
    if 'id' not in m.columns:
        m.insert(0, 'id', ids)
    value_vars = [i for i in m.columns.values[1:]]
    return pd.melt(m, id_vars=['id'], value_vars=value_vars)

def get_gl_preds(m, verbosity=False):
    melted = get_melted(m)
    full_sf = gl.SFrame(melted)
    train_sf = gl.SFrame(melted.dropna())
    gl_rec = gl.factorization_recommender.create(train_sf, user_id='id', item_id='variable', target='value', verbose=verbosity)
    predictions = pd.DataFrame(np.array(gl_rec.predict(full_sf)), columns=['prediction'])
    return predictions

def get_gl_recommender(m, verbosity=False):
    melted=get_melted(m)
    return gl.factorization_recommender.create(gl.SFrame(melted), user_id='id', item_id='variable', target='value', verbose=verbosity)


def get_coeffs(rec):
    '''
    Takes a graphlab recommender and returns:
    user_intercept
    user_factors
    item_intercept
    item_factors
    intercept
    '''
    user_intercept = rec.coefficients['id']['linear_terms']
    user_factors = rec.coefficients['id']['factors']
    item_intercept = rec.coefficients['variable']['linear_terms']
    item_factors = rec.coefficients['variable']['factors']
    intercept = rec.coefficients['intercept']
    return user_intercept, user_factors, item_intercept, item_factors, intercept

def gl_predictor(user_rec, item_rec, user_id):
    '''
    Takes a graphlab recommender for the user domain and one for the item domain and the user id
    returns a vector of the item predictions for the user
    '''
    user_intercept, user_factors, _, _, u_intercept = get_coeffs(user_rec)
    _, _, item_intercept, item_factors, i_intercept = get_coeffs(item_rec)

    mat = np.dot(np.array(user_factors), np.array(item_factors).T)
    user_vec = mat[user_id] + item_intercept
    user_vec = user_vec + user_intercept[user_id]
    user_vec = user_vec + np.mean([u_intercept, i_intercept])
    return user_vec




if __name__ == '__main__':
    with open('data/actual_movies.pkl') as f:
        movies_df = pickle.load(f)
    with open('data/actual_songs.pkl') as f:
        songs_df = pickle.load(f)

    movies = Matrix(movies_df)
    songs = Matrix(songs_df)
    song_predictions = PredictedMatrix(movies, songs)
    test = Tester(songs, song_predictions)
    print test.mse
    '''
    Find MSE of Block Diagonal NaN Matrix
    '''
    bd, bd_matrix, full = block_diagonal(movies, songs)
    
    #bd_U, bd_s, bd_V = svd(bd.matrix, full_matrices=False)
    #bd_filled = NuclearNormMinimization().complete(bd)
    #bd_filled = SoftImpute().complete(bd)
    #bd_filled = KNN(k=3).complete(bd)
    #bd_filled = IterativeSVD.complete(bd)
    #bd_df = pd.DataFrame(bd_filled, index=full.matrix.index, columns=full.matrix.columns)
    #bd_filled = KNN(k=3).complete(bd)
    #bd_filled = BiScaler().complete(bd)
    #bd_pred = np.dot(bd_U, np.dot(np.diag(bd_s), bd_V))
    #bd_pred = PredictedMatrix(bd, bd)
    #bd_pred = Matrix(bd_df)
    #test2 = Tester(full, bd_pred)
    #test2 = Tester(full, Matrix(bd))
    #t_mse = test2.mse

    '''
    build matrix out of full and predicted parts
    '''
    m1_matrix = subset_matrix(movies, idx=None)
    s2_matrix = subset_matrix(songs, idx=None, how='bottom')
    pred_s1 = PredictedMatrix(m1_matrix, s2_matrix)
    pred_full = build_full_predicted_matrix(m1_matrix, s2_matrix)
    test3 = Tester(full, pred_full)

    #test interleaved matrix
    interleaved = build_interleaved_matrix(movies, songs, 'zero')
    test4 = Tester(full, interleaved)
    print test4.mse
    
    #test graphlab's predictions on block diagonal NaN matrix
    gl_preds = get_gl_preds(bd)
    full_melt = get_melted(full.matrix)
    full_preds = pd.concat([full_melt, gl_preds], axis=1)
    error = np.mean(np.square(full_preds.eval('value-prediction')))
    
    '''
    incorporate gl factors
    '''
    
    

    eighties_rec = get_gl_recommender(songs.matrix.iloc[:, 10:].copy())
    current_rec = get_gl_recommender(songs.matrix.iloc[:, :10].copy())
    songs_rec = get_gl_recommender(songs.matrix.copy())
    movies_rec = get_gl_recommender(movies.matrix.copy())
    #eighties_ui = eighties_rec.coefficients['id']['linear_terms']
    #eighties_ufactors = eighties_rec.coefficients['id']['factors']
    #eighties_ii = eighties_rec.coefficients['variable']['linear_terms']
    #eighties_ifactors = eighties_rec.coefficients['variable']['factors']
    #eighties_intercept = eighties_rec.coefficients['intercept']

    #eighties_ui, eighties_ufactors, eighties_ii, eighties_ifactors, eighties_intercept = get_coeffs(eighties_rec)
    song_ui, song_ufactors, song_ii, song_ifactors, song_i = get_coeffs(songs_rec)
    movie_ui, movie_ufactors, movie_ii, movies_ifactors, movie_i = get_coeffs(movies_rec)




    
    '''
    code for clustering songs and movies
    '''
    movies_factors = movies.V.T
    songs_factors=songs.V.T
    m_factor_df = pd.DataFrame(movies_factors, index=movies.items).rename(columns=lambda x: 'factor_'+str(x))
    s_factor_df = pd.DataFrame(songs_factors, index=songs.items).rename(columns=lambda x: 'factor_'+str(x))
    cluster_df = pd.concat([m_factor_df, s_factor_df])
    cluster_df.to_pickle('data/cluster_df.pkl') #for flask_app to read
    n_clusters = 4
    clust = AC(affinity='cosine', n_clusters=n_clusters, linkage='average')
    clusters = clust.fit(cluster_df)

    node_list = [{"id":cluster_df.index[x], "group":clusters.labels_[x]} for x in xrange(cluster_df.shape[0])]
    link_list = []
    dists = pairwise_distances(cluster_df, metric='euclidean')
    items=[x for x in cluster_df.index]
    for idx, n in enumerate(items):
        if idx < len(items):
            targets = [item for item in items[idx+1:]]
            for idx2, n2 in enumerate(targets):
                #value = int(round(dists[idx, idx2],1)*10)
                value = dists[idx, idx2]
                link_list.append({"source":n, "target":n2, "value":value})

    force_dict = {"nodes":node_list, "links":link_list}
    with open('app/static/clusters.json', 'w') as f:
        json.dump(force_dict, f)
            
    km = KMeans(n_clusters = 3)
    km.fit(cluster_df)
    movies_mean_df = pd.DataFrame(np.mean(movies.matrix.copy()))
    songs_mean_df = pd.DataFrame(np.mean(songs.matrix.copy()))
    mean_ratings = pd.concat([movies_mean_df, songs_mean_df]).rename(columns={0:'rating'})
    mean_ratings['cluster'] = km.labels_
    cluster_list = {}
    node_list = []
    for idx, name in enumerate(mean_ratings.index):
        #node = {"id":name, "rating":round(mean_ratings.loc[name]['rating'],2), "cluster":int(mean_ratings.loc[name]['cluster'])}
        node = {"rating":round(mean_ratings.loc[name]['rating'],2), "cluster":int(mean_ratings.loc[name]['cluster'])}
        node_list.append(node)
    cluster_list = {"nodes":node_list}
    with open('app/static/cluster2.json', 'w') as f:
        json.dump(cluster_list, f)
        
