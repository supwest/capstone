from flask import Flask


app = Flask(__name__)





@app.route('/')
@app.route('/home')
def index():
    #return render_template('index.html')
    return str('hello')

@app.route('/recommender')
def recommender():
    return "recommender goes here"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
