from flask import Blueprint, jsonify

main = Blueprint('main', __name__)

@main.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': 'online',
        'message': 'Rock Fluid Monitor API is running'
    }) 