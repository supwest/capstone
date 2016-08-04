from flask import Flask, render_template, request, url_for, jsonify
import cPickle as pickle
import numpy as np
import graphlab as gl
import pandas as pd
from sklearn.cluster import AgglomerativeClustering as AC, DBSCAN
import json
app = Flask(__name__)

movie_dict={'inside_out':'Inside Out',
            'get_hard':'Get Hard',
            'star_wars':'Star Wars: The Force Awakens',
            'fast_7':'Fast 7',
            'the_fault_in_our_stars':'The Fault in Our Stars',
            'the_martian':'The Martian',
            'the_lego_movie':'The Lego Movie',
            'the_revenant':'The Revenant',
            'ted_2':'Ted 2',
            'mad_max_fury_road':'Mad Max: Fury Road'
            }

movie_order_dict={'inside_out':9,
            'get_hard':0,
            'star_wars':2,
            'fast_7':7,
            'the_fault_in_our_stars':3,
            'the_martian':1,
            'the_lego_movie':5,
            'the_revenant':4,
            'ted_2':6,
            'mad_max_fury_road':8
            }

song_titles_dict = {'bad_blood':['Bad Blood', 'Taylor Swift',''], 
                    'lets_dance':["Let's Dance", "David Bowie",''],
                    'hotline_bling':["Hotline Bling", "Drake",''],
                    "panda":["Panda", "Desiigner"],
                    "stressed_out":["Stressed Out", "Twenty One Pilots",''],
                    "rock_me_amadeus":["Rock Me Amadeus", "Falco",''],
                    "dont_let_me_down":["Don't Let Me Down", "The Chainsmokers"],
                    "hands_to_myself":["Hands to Myself", "Selena Gomez",''],
                    "sorry":["Sorry", "Justin Beiber",''],
                    "fancy":["Fancy", "Iggy Azalea",''],
                    "work_from_home":["Work From Home", "Fifth Harmony",''],
                    "trap_queen":["Trap Queen", "Fetty Wap",''],
                    "come_on_eileen":["Come on Eileen", "Dexy's Midnight Runenrs",''],
                    "when_doves_cry":["When Doves Cry", "Prince",''],
                    "sweet_child_o_mine":["Sweet Child O' Mine", "Guns N' Roses",''],
                    "billie_jean":["Billie Jean", "Michael Jackson",'<iframe style="width:120px;height:240px;" marginwidth="0" marginheight="0" scrolling="no" frameborder="0" src="//ws-na.amazon-adsystem.com/widgets/q?ServiceVersion=20070822&OneJS=1&Operation=GetAdHtml&MarketPlace=US&source=ac&ref=tf_til&ad_type=product_link&tracking_id=culwes-20&marketplace=amazon&region=US&placement=B0013DA95O&asins=B0013DA95O&linkId=8570d31fd5df8b048300096c3b917c1f&show_border=false&link_opens_in_new_window=false&price_color=333333&title_color=0066c0&bg_color=ffffff"></iframe>'],
                    "every_breath_you_take":["Every Breath You Take", "The Police",''],
                    "call_me":["Call Me", "Blondie",''],
                    "another_one_bites_the_dust":["Another One Bites the Dust", "Queen",''],
                    "centerfold":["Centerfold", "J. Geils Band",'']}


def get_movies(song):
    #return str(song)
    song_dict = {'bad_blood':0, 'stressed_out':1, 'panda':5, 'rock_me_amadeus':10, 'dont_let_me_down':2, 'hotline_bling':3, 'hands_to_myself':4, 'work_from_home':6, 'trap_queen':7, 'sorry':8, 'fancy':9, 'come_on_eileen':11, 'when_doves_cry':12, 'sweet_child_o_mine':13, 'billie_jean':14, 'every_breath_you_take':15, 'call_me':16, 'another_one_bites_the_dust':17, 'centerfold':18, 'lets_dance':19}
    path_to_movie_pickle='/home/ubuntu/capstone/data/movie_titles.pkl'
    path_to_sim_pickle='/home/ubuntu/capstone/data/similarity.pkl'
    with open(path_to_movie_pickle) as f:
        movie_titles = pickle.load(f)
    #movies_titles = pickle.load(path_to_movie_pickle)
    with open(path_to_sim_pickle) as f:
        similarity = pickle.load(f)
    images_dict = {'The Lego Movie': 'https://upload.wikimedia.org/wikipedia/en/1/10/The_Lego_Movie_poster.jpg', 
                'Mad Max: Fury Road':'https://upload.wikimedia.org/wikipedia/en/6/6e/Mad_Max_Fury_Road.jpg',
                'The Fault in Our Stars':'https://upload.wikimedia.org/wikipedia/en/4/41/The_Fault_in_Our_Stars_%28Official_Film_Poster%29.png',
                'Get Hard': 'https://upload.wikimedia.org/wikipedia/en/3/38/Get_Hard_film_poster.png',
                'Inside Out': 'https://upload.wikimedia.org/wikipedia/en/0/0a/Inside_Out_%282015_film%29_poster.jpg',
                'The Martian': 'https://upload.wikimedia.org/wikipedia/en/c/cd/The_Martian_film_poster.jpg',
                'Ted 2': 'https://upload.wikimedia.org/wikipedia/en/2/24/Ted_2_poster.jpg',
                'Fast 7': 'https://upload.wikimedia.org/wikipedia/en/b/b8/Furious_7_poster.jpg',
                'The Revenant':'https://upload.wikimedia.org/wikipedia/en/b/b6/The_Revenant_2015_film_poster.jpg',
                'Star Wars The Force Awakens': 'https://upload.wikimedia.org/wikipedia/en/a/a2/Star_Wars_The_Force_Awakens_Theatrical_Poster.jpg'
                }
    #similarity = pickle.load(path_to_sim_pickle)
    movies_list = movie_titles[np.argsort(similarity[song_dict[song]])]
    images_list = [images_dict[movie] for movie in movies_list]
    
    return movies_list, images_list

def get_movie_recommended_songs(movie):
    path_to_songs = '/home/ubuntu/capstone/data/song_titles.pkl'
    path_to_matrix = '/home/ubuntu/capstone/data/movie_song_matrix.pkl'

    movie_dict={'inside_out':9,
            'get_hard':0,
            'star_wars':2,
            'fast_7':7,
            'the_fault_in_our_stars':3,
            'the_martian':1,
            'the_lego_movie':5,
            'the_revenant':4,
            'ted_2':6,
            'mad_max_fury_road':8
            } 

    with open(path_to_songs) as f:
        song_titles = pickle.load(f)
    with open(path_to_matrix) as f:
        rec_matrix = pickle.load(f)

    song_recs = song_titles[np.argsort(rec_matrix[movie_dict[movie]])[::-1]]
    return song_recs

def get_rec_coeffs(rec):
    user_intercept = rec.coefficients['id']['linear_terms']
    user_factors = rec.coefficients['id']['factors']
    item_intercept = rec.coefficients['variable']['linear_terms']
    item_factors = rec.coefficients['variable']['factors']
    intercept = rec.coefficients['intercept']
    return user_intercept, user_factors, item_intercept, item_factors, intercept

def get_song_recs(ratings, n_features):
    path_to_songs_sf = '/home/ubuntu/capstone/data/flask_songs_sf'
    path_to_movies_sf = '/home/ubuntu/capstone/data/flask_movies_sf'
    songs_sf = gl.load_sframe(path_to_songs_sf)
    songs_df = songs_sf.to_dataframe()
    value_vars = [x for x in songs_df.columns if x != 'id']
    ids = [x for x in songs_df.index]
    if 'id' not in songs_df.columns:
        songs_df.insert(0, 'id', ids)
    songs_melted = gl.SFrame(pd.melt(songs_df, id_vars = 'id', value_vars=value_vars))
    songs_rec = gl.factorization_recommender.create(songs_melted, user_id = 'id', item_id='variable', target='value', num_factors = n_features)
    _, _, songs_item_intercept, songs_item_factors, songs_intercept = get_rec_coeffs(songs_rec)
    movies_sf = gl.load_sframe(path_to_movies_sf)
    movies_df = movies_sf.to_dataframe()
    
    value_vars = [x for x in movies_df.columns if x != 'id']
    #new_ratings = [int(x) if x != '9' else np.nan for x in ratings]

    new_ratings = {movie_dict[name]:int(ratings[name]) for name in ratings}
    new_df = pd.DataFrame.from_dict(new_ratings, orient='index').replace(-1,np.nan)
    #new_ratings = np.array(new_ratings).reshape(10,1)
    #new_df = pd.DataFrame([new_ratings], dtype='float')
    #new_df.columns = movies_df.columns
    movies_df = pd.concat([movies_df, new_df])
    ids = [str(i) for i in movies_df.index]
    #if 'id' not in songs_df.columns:
    movies_df.insert(0, 'id', ids)
    movies_melted = gl.SFrame(pd.melt(movies_df, id_vars='id', value_vars=value_vars)).dropna()
    movies_rec = gl.factorization_recommender.create(movies_melted, user_id='id', item_id='variable', target='value', num_factors=n_features)
    movies_user_intercept, movies_user_factors, _, _, movies_intercept = get_rec_coeffs(movies_rec)
    comb = np.dot(np.array(movies_user_factors)[-1], np.array(songs_item_factors).T)
    #comb = comb + songs_item_intercept
    #comb = comb + movies_user_intercept[0]
    #comb = comb + np.mean([movies_intercept, songs_intercept])
    return songs_df.columns[1:][np.argsort(comb)[::-1]]
    #return comb

def get_wine_recs(ratings):
    path_to_movies = '/home/ubuntu/capstone/data/flask_movies_sf'
    path_to_wine = '/home/ubuntu/capstone/data/gridsearch_sf'
    wine_rec = gl.load_model(path_to_wine)
    movies_sf = gl.load_sframe(path_to_movies)
    movies_df = movies_sf.to_dataframe()
    value_vars = [x for x in movies_df.columns if x != 'id']
    new_ratings = {movie_dict[name]:int(ratings[name]) for name in ratings}
    new_df = pd.DataFrame.from_dict([new_ratings], orient='columns').replace(-1, np.nan)
    movies_df = pd.concat([movies_df, new_df]).reset_index(drop=True)
    ids = [i for i in movies_df.index]
    movies_df.insert(0, 'id', ids)
    movies_melted = gl.SFrame(pd.melt(movies_df, id_vars='id', value_vars=value_vars)).dropna()
    movies_rec = gl.factorization_recommender.create(movies_melted, user_id = 'id', item_id='variable', target='value')
    movies_user_intercept, movies_user_factors, _, _, movies_intercept = get_rec_coeffs(movies_rec)
    #_, _, wine_item_intercept, wine_item_factors, wine_intercept = get_rec_coeffs(wine_rec)
    wine_item_factors = np.array(wine_rec.coefficients['wine_name']['factors'])[:,:8]
    wine_names = np.array(wine_rec.coefficients['wine_name']['wine_name'])
    comb = np.dot(np.array(movies_user_factors[-1]), wine_item_factors.T)
    return wine_names[np.argsort(comb)[::-1]]

def get_wines_for_movie(movie):
    path_to_wine = '/home/ubuntu/capstone/data/gridsearch_sf'
    path_to_movies = '/home/ubuntu/capstone/data/flask_movies_sf'
    wine_rec = gl.load_model(path_to_wine)
    movies_sf = gl.load_sframe(path_to_movies)
    cols = movies_sf.column_names()
    #movies_sf = movies_sf.add_row_number()
    movies_df = movies_sf.to_dataframe()
    ids = [i for i in movies_df.index]
    movies_df.insert(0, 'id', ids)
    value_vars = [x for x in movies_df.columns if x != 'id']
    movies_melted = gl.SFrame(pd.melt(movies_df, id_vars='id', value_vars=value_vars)).dropna()
    movies_rec = gl.factorization_recommender.create(movies_melted, user_id='id', item_id='variable', target='value')
    movie_pos = movie_order_dict[movie]
    sims = pairwise_distances(np.array(movies_rec.coefficients['variable']['factors'])[movie_pos].reshape(1,-1), np.array(wine_rec.coefficients['wine_name']['factors'])[:,:8], metric='cosine')
    wine_names = np.array(wine_rec.coefficients['wine_name']['wine_name'])
    return wine_names[np.argsort(sims[0])[::-1]][:5]


def get_data():
    with open('static/clusters.json', 'r') as f:
        g = f.read()
    #graph_dict = json.loads('static/clusters.json')
    #RESULTS = {'children': []}
    #for k in graph_dict:
    #    RESULTS['children'].append({graph_dict[k]})
    return g

def get_data_2():
    with open('static/cluster2.json', 'r') as f:
        g = f.read()
    return g
    
@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/get_movie', methods=['GET','POST'])
def recommender():
    if request.method == 'POST':
        song = (request.form['song'])
        recommendations, images = get_movies(song)
        return render_template('get_movie_recs.html', movie1=recommendations[0], image1=images[0], movie2=recommendations[1], image2=images[1], movie3=recommendations[2], image3=images[2], song=song_titles_dict[song])
    return render_template('get_movie.html')

@app.route('/get_song', methods=["GET",'POST'])
def movie_song_recs():
    movie_dict={'inside_out':'Inside Out',
            'get_hard':'Get Hard',
            'star_wars':'Star Wars: The Force Awakens',
            'fast_7':'Fast 7',
            'the_fault_in_our_stars':'The Fault in Our Stars',
            'the_martian':'The Martian',
            'the_lego_movie':'The Lego Movie',
            'the_revenant':'The Revenant',
            'ted_2':'Ted 2',
            'mad_max_fury_road':'Mad Max: Fury Road'
            }
    if request.method == 'POST':
        movie = (request.form['movie'])
        recommendations = get_movie_recommended_songs(movie)
        return render_template('get_song_recs.html',movie=movie_dict[movie], songs=recommendations[:5], movies=movie_dict)
    return render_template('get_song.html', movies=movie_dict)


@app.route('/get_song2', methods=['GET','POST'])
def get_updated_song_recs():
    if request.method == 'POST':
        ratings = ({m:request.form[m] for m in movie_dict})
        #n_features = request.form['n_features']
        n_features = 8
        recommendations = get_song_recs(ratings, n_features)
        #ratings = {}
        #for m in movie_list:
        #    ratings[m] = (request.form[m])
        #recommendations = get_song_recommendations(rating)
        return render_template('get_song_recs_2.html', songs=recommendations[:5])
    return render_template('get_song2.html', movies=movie_dict, n_feats = [n+1 for n in xrange(8)], nums = [str(n) for n in xrange(8)])

@app.route('/get_wine', methods = ["GET","POST"])
def get_wine():
    if request.method == 'POST':
        ratings = ({m:request.form[m] for m in movie_dict})
        recommendations = get_wine_recs(ratings)
        recs = [r.decode('utf-8', 'ignore') for r in recommendations]
        return render_template('get_wine_recs.html', wines=recs[:5])
    return render_template('get_wine.html', movies=movie_dict)

@app.route('/get_wine2', methods=["GET","POST"])
def get_wine_2():
    movie_dict={'inside_out':'Inside Out',
            'get_hard':'Get Hard',
            'star_wars':'Star Wars: The Force Awakens',
            'fast_7':'Fast 7',
            'the_fault_in_our_stars':'The Fault in Our Stars',
            'the_martian':'The Martian',
            'the_lego_movie':'The Lego Movie',
            'the_revenant':'The Revenant',
            'ted_2':'Ted 2',
            'mad_max_fury_road':'Mad Max: Fury Road'
            }

    if request.method == 'POST':
        movie = (request.form['movie'])
        recommendations = get_wines_for_movie(movie)
        recs = [r.decode('utf-8', 'ignore') for r in recommendations]
        return render_template('get_wine_recs_2.html', wines = recs, movie=movie_dict[movie])
    return render_template('get_wine2.html', movies=movie_dict)



@app.route('/clusters')
def show_clusters():
    #cluster_df = pd.read_pickle('/home/cully/Documents/capstone/data/cluster_df.pkl')
    #clust = AC(n_clusters=4, affinity='cosine', linkage='average')
    #clusters=clust.fit(cluster_df)
    j = {}
    #for x in xrange(4):
    #    j[x] = cluster_df.index[clusters.labels_==x]
    #    print cluster_df.index[clusters.labels_==x]
    #json.dumps(j)
    return render_template('clusters.html')

@app.route('/data')
def data():
    return get_data()
    #return jsonify(get_data())

@app.route('/clusters2')
def show_clusters2():
    return render_template('clusters2.html')

@app.route('/data2')
def data_2():
    return get_data_2()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
