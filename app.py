import pickle
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app) 

# Load pre-trained models and data
model = pickle.load(open('pickles/model.pkl', 'rb'))
book_names = pickle.load(open('pickles/book_names.pkl', 'rb'))
final_rating = pickle.load(open('pickles/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('pickles/book_pivot.pkl', 'rb'))

def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []
    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])
    for name in book_name[0]:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)
    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)
    return poster_url

def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    poster_url = fetch_poster(suggestion)
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)
    return books_list[1:], poster_url[1:]

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    book_name = request.args.get('book')
    if book_name not in book_names:
        return jsonify({'error': 'Book not found'}), 404
    recommended_books, poster_url = recommend_book(book_name)
    recommendations = [{'book': book, 'poster_url': url} for book, url in zip(recommended_books, poster_url)]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)