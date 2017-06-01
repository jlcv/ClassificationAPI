#!flask/bin/python
from flask import Flask, jsonify, request
import json
from post_new import post_new_value
from training import get_training_results
from testing import get_testing_results
from predict import post_prediction

app = Flask(__name__)

base_api = "/classification_api/v1"
api_key = "d8ccc900-7c92-4722-be58-5b747184dcb7"

def check_api_key(obtained_api_key):
    if obtained_api_key == api_key:
        return True
    else:
        return False
    

@app.route(base_api + '/predict', methods=['POST'])
def post_predict():
    if check_api_key(obtained_api_key=request.headers.get('X-Api-Key')):
        try:
            data = request.json
            return jsonify({'success': True, 'prediction': json.loads(post_prediction(body=data['body'], project_name=data['project_name']))})
        except Exception, e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Unrecognized Api Key'}), 400

@app.route(base_api + '/new', methods=['POST'])
def post_new():
    if check_api_key(obtained_api_key=request.headers.get('X-Api-Key')):
        try:
            data = request.json
            return jsonify({'success': post_new_value(values=data['values'], category=data['category'], project_name=data['project_name']), 'result': json.loads(get_training_results(project_name=data['project_name']))})
        except Exception, e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Unrecognized Api Key'}), 400

@app.route(base_api + '/testing', methods=['POST'])
def post_test_results():
    if check_api_key(obtained_api_key=request.headers.get('X-Api-Key')):
        try:
            data = request.json
            return jsonify({'success': True, 'result': json.loads(get_testing_results(project_name=data['project_name']))})
        except Exception, e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Unrecognized Api Key'}), 400

@app.route(base_api + '/training', methods=['POST'])
def post_train_results():
    if check_api_key(obtained_api_key=request.headers.get('X-Api-Key')):
        try:
            data = request.json
            return jsonify({'success': True, 'result': json.loads(get_training_results(project_name=data['project_name']))})
        except Exception, e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Unrecognized Api Key'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0')
