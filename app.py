#!flask/bin/python
from flask import Flask, jsonify, request
import json
from email_post_new import post_new_email
from email_training import get_training_results
from email_predict import post_prediction

app = Flask(__name__)

base_api = "/classification_api/v1"
api_key = "d8ccc900-7c92-4722-be58-5b747184dcb7"

def check_api_key(obtained_api_key):
    if obtained_api_key == api_key:
        return True
    else:
        return False
    

@app.route(base_api + '/predict', methods=['POST'])
def get_tasks():
    if check_api_key(obtained_api_key=request.headers.get('X-Api-Key')):
        try:
            data = request.json
            return jsonify({'success': True, 'prediction': json.loads(post_prediction(email_body=data['email_body'], categories=data['categories']))})
        except Exception, e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Unrecognized Api Key'}), 400

@app.route(base_api + '/new/email', methods=['POST'])
def post_email():
    if check_api_key(obtained_api_key=request.headers.get('X-Api-Key')):
        try:
            data = request.json
            return jsonify({'success': post_new_email(email_body=data['email_body'], category=data['category'])})
        except Exception, e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Unrecognized Api Key'}), 400

@app.route(base_api + '/testing', methods=['POST'])
def get_test_results():
    if check_api_key(obtained_api_key=request.headers.get('X-Api-Key')):
        try:
            data = request.json
            return jsonify({'success': True, 'result': json.loads(get_training_results(categories=data['categories']))})
        except Exception, e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Unrecognized Api Key'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0')
