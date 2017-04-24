#!flask/bin/python
from flask import Flask, jsonify, request
import json
from email_post_new import post_new_email
from email_training import get_training_results
from email_predict import post_prediction

app = Flask(__name__)

base_api = "/classification_api/v1"

@app.route(base_api + '/predict', methods=['POST'])
def get_tasks():
    try:
        data = request.json
        return jsonify({'success': True, 'prediction': json.loads(post_prediction(email_body=data['email_body'], categories=data['categories']))})
    except Exception, e:
        return jsonify({'error': str(e)}), 400

@app.route(base_api + '/new/email', methods=['POST'])
def post_email():
    try:
        data = request.json
        return jsonify({'success': post_new_email(email_body=data['email_body'], category=data['category'])})
    except Exception, e:
        return jsonify({'error': str(e)}), 400

@app.route(base_api + '/testing', methods=['POST'])
def get_test_results():
    try:
        data = request.json
        return jsonify({'success': True, 'result': json.loads(get_training_results(categories=data['categories']))})
    except Exception, e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0')
