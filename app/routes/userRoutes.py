from flask import Blueprint, request, jsonify
from app.controllers.authController import register, login

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

# def register_auth_routes(app):
#     """
#     Register authentication routes with the application
#     """
#     app.register_blueprint(auth_bp)