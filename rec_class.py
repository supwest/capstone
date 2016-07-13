import pandas as pd
import graphlab as gl
import numpy as np



def CCR(object):
    '''
    Cross Class Recommeder:
    Should take two dataframes (or maybe graphlab recommeders)
    and use the decomposed matrices to generate new recommendations

    '''

    def __init__(self, user_target, domain_target):
        '''
        user_target is the ratings matrix which contains the target user
        domain_target is the ratings matrix made up from the domain target
        user_U is the left matrix from the decomp of user_target
        domain_V is the right matrix from the decomp of domain_target

        '''

        self.user_target = user_target
        self.domain_target = domain_target
        self.user_U = None
        self.domain_V = None


    def _get_user_factors(self):
        '''
        this method will decompose the user matrix and return the user factor loadings
        '''
        self.U, s, V = np.linalg.svd(self.user_target)

    def _get_domain_factors(self):
        '''
        this method will decompose the domain matrix and return the domain factor loadings

        '''

        U, s, self.V = np.linalg.svd(self.domain_target)
        #return modified_V

    

    def get_recs(self, user_id):
        '''
        get recommednations for user
        '''

        new_mat = np.dot(self.user_U, np.transpose(self.domain_V))
        return new_mat[user_id]
