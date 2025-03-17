from flask import Blueprint, request, jsonify
from app.controllers.user_controller import register, login

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/register', methods=['POST'])
def register_route():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400
    return register()

@auth_bp.route('/login', methods=['POST'])
def login_route():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400
    return login()

def register_auth_routes(app):
    app.register_blueprint(auth_bp)