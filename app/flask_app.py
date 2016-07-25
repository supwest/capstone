from flask import Flask, render_template, request, url_for
import cPickle as pickle
import numpy as np

app = Flask(__name__)

def get_movies(song):
    #return str(song)
    song_dict = {'bad_blood':0, 'stressed_out':1, 'panda':5, 'rock_me_amadeus':10, 'dont_let_me_down':2, 'hotline_bling':3, 'hands_to_myself':4, 'work_from_home':6, 'trap_queen':7, 'sorry':8, 'fancy':9, 'come_on_eileen':11, 'when_doves_cry':12, 'sweet_child_o_mine':13, 'billie_jean':14, 'every_breath_you_take':15, 'call_me':16, 'another_one_bites_the_dust':17, 'centerfold':18, 'lets_dance':19}
    path_to_movie_pickle='/home/cully/Documents/capstone/data/movie_titles.pkl'
    path_to_sim_pickle='/home/cully/Documents/capstone/data/similarity.pkl'
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

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')
    #return str('hello')

@app.route('/home', methods=['GET','POST'])
def recommender():
    #return "recommender goes here"
    if request.method == 'POST':
        song = (request.form['song'])
        recommendations, images = get_movies(song)
        return render_template('get_movie_recs.html', movie1=recommendations[0], image1=images[0], movie2=recommendations[1], image2=images[1], movie3=recommendations[2], image3=images[2])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
