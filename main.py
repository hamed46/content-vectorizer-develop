from flask import Flask, jsonify, request
from flask_restful import Api
from flask_cors import CORS
from service.generate_vector import get_embeddings

app = Flask(__name__)
CORS(app)
api = Api(app)


@app.route('/api/v1/embeddings', methods=['POST'])
def handle_post_request():
    # Handle the POST request here
    data = request.get_json()
    return jsonify(get_embeddings(data))


if __name__ == "__main__":
    app.run(debug=True, port=5000, host='localhost')
