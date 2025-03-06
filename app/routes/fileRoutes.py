from flask import Blueprint, request, jsonify
from app.controllers.fileController import upload_file

file_bp = Blueprint('file', __name__, url_prefix="/api/files")

@file_bp.route('/upload', methods=['POST'])
def upload_file_route():
    if not request.files:
        return jsonify({"msg": "Missing file in request"}), 400
    return upload_file()

def register_file_routes(app):
    app.register_blueprint(file_bp)
