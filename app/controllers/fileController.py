import os
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename

from app.utils.fileUtils import allowed_file

def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    allowed_extensions = request.form.get('allowed_extensions',None)
    if allowed_extensions:
        allowed_extensions = set(allowed_extensions.split(','))

    if file and allowed_file(file.filename, allowed_extensions):
        filename = secure_filename(file.filename)
        unique_filename = f"{filename}"

        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, unique_filename)
        file.save(file_path)

        file_size = os.path.getsize(file_path)

        return jsonify({
            "msg": "File uploaded successfully",
            "filename": unique_filename,
            "original_filename": filename,
            "size": file_size,
            "path": file_path
        }), 201

    return jsonify({"msg": "File type not allowed"}), 400
