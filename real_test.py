
'''
script to test the movies and song matrices from the google forms survey
'''
import pandas as pd
import numpy as np
from numpy.linalg import svd
import cPickle as pickle

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
        return Matrix(self.matrix[:idx]), Matrix(self.matrix[idx:])

    
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
        return Matrix(self.matrix[:idx]), Matrix(self.matrix[idx:])

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

    def _calc_mse(self, matrix1, matrix2):
        '''
        Input: df
        Output: df

        takes a 'true matrix' and returns the MSE of the difference
        between the true matrix and the fitted matrix

        '''
        ea = matrix1.reset_index(drop=True) -matrix2.reset_index(drop=True)
        #print ea
        #sq_e = e**2
        #sum_sq_e = np.mean(np.sum(sq_e))
        #self.error_ = sum_sq_e
        return np.mean(np.sum(np.square(ea)))


    def common_user_test(self, n_common=1):
        '''
        Somethings wrong in here.
        fixed - The mse call returns the same thing before and after changing
        need to use copies so doesn't really change underlying matrices
        (run common_user_test with different nums in common to see problem)
        '''
        m1, m2 = self.matrix_1.split()
        m1_temp = Matrix(m1.matrix.copy())
        #print m1.matrix
        #print m2.matrix
        s1, s2 = self.matrix_2.split()
        #m1.reset_index()
        #s2.reset_index()
        print "M1", m1.matrix
        pred_mat = PredictedMatrix(m1_temp,s2)
        #print pred_mat.matrix
        t1 =  Tester(s2, pred_mat)
        mse_1 = t1.mse
        #print "mse1 is {}".format( mse_1)
        for idx in xrange(n_common):
            #print m1.matrix.iloc[idx]
            m1_temp.matrix.iloc[idx] = m2.matrix.iloc[idx]
            #print m1.matrix.iloc[idx]
            #raw_input("w")
        new_m1 = Matrix(m1_temp.matrix)
        #print "m1 after", m1.matrix
        pred_mat2 = PredictedMatrix(new_m1, s2)
        #print pred_mat2.matrix
        t2 = Tester(s2, pred_mat2)
        mse_2 = t2.mse
        print "Mse2 is {}".format(mse_2)
        print "mse change is {}".format(mse_2-mse_1)
        return mse_2-mse_1
    


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

