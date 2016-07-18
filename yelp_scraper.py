import urllib
import requests
import pymongo
from pymongo import MongoClient
from pymongo import errors
import os

'''
will need to install yelp-python from yelp github
'''

from yelp.client import Client as YelpClient
from yelp.oauth1_authenticator import Oauth1Authenticator
class YelpScraper(object):
    '''
    Uses Yelp API to gather reviews and insert them into a database.
    Will (eventually) take an argument to use Yelp's Search API to get 
    businesses or to use Yelp's Business API to get reviews

    '''
    def __init__(self):
        self.mongo_client = None
        self.yelp_client = None
        pass

    def set_yelp_client(self, auth):
        '''
        initializes yelp search client. 
        auth is a yelp Oauth1Authenticator instance
        '''
        self.yelp_client = YelpClient(auth)

    def set_mongo_client(self, db_name, coll_name):
        '''
        initializes a mongo_client for working with mongodb database.
        db_name should be yelp.
        coll_name should be either business to store businesses or 
        reviews to store the reviews for businesses in the business coll.
        '''
        self.mongo_client = MongoClient()
        self.db = self.mongo_client[db_name]
        self.coll = self.db[coll_name]

    def scrape(self, loc, params):
        '''
        This should do all the scraping by calling the search client, then inserting the results into the database.
        '''
        _offset = 0
        search = self._make_yelp_search(loc, params)


    def _make_yelp_search(self, loc, params):
        '''
        called by scrape to get search results
        '''
        self.yelp_client.search(loc, **params)




if __name__ == '__main__':
    yelp = YelpScraper()

    auth = Oauth1Authenticator(
        token = os.environ['YELP_TOKEN'],
        token_secret = os.environ['TOKEN_SECRET'],
        consumer_key = os.environ['YELP_KEY'],
        consumer_secret = os.environ['SECRET_KEY']
    )

    yelp.set_yelp_client(auth)
    #yelp.set_mongo_client('yelp', 'reviews')
    
    params = {'term': 'food'}
    loc = 'Denver'

    yelp.scrape(loc, params)


