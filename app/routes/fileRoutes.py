from flask import Blueprint, request, jsonify
from app.controllers.fileController import uploadFile

file_bp = Blueprint('file',__name__,url_prefix="/api/file")

@file_bp.route('/upload', methods=['POST'])
def upload_file_route():
    if not request.files:
        return jsonify({"msg":"Missing file in request"}), 400
    return uploadFile()