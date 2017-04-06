#!flask/bin/python
from flask import Flask, jsonify, request
import json
import email_post_new
from email_training import get_training_results
from email_predict import post_prediction

app = Flask(__name__)

base_api = "/classification_api/v1"

@app.route(base_api + '/predict', methods=['POST'])
def get_tasks():
    try:
        data = request.json
        return jsonify({'success': post_prediction(email_body=data['email_body'])})
    except:
        return jsonify({'error': "Invalid parameters."}), 400

@app.route(base_api + '/new/email', methods=['POST'])
def post_email():
    try:
        data = request.json
        return jsonify({'success': email_post_new.post_email(email_body=data['email_body'])})
    except:
        return jsonify({'error': "Invalid parameters."}), 400

@app.route(base_api + '/testing', methods=['GET'])
def get_test_results():
    return jsonify({'result':get_training_results()})

if __name__ == '__main__':
    app.run(debug=True)
