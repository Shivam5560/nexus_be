import os
from werkzeug.utils import secure_filename
from app.models.resume_model import Resume
from app.services.db_service import get_db
from app.services.auth_service import get_user_by_id
from bson import ObjectId
from flask import current_app


def allowed_file(filename, allowed_extensions=None):
    if allowed_extensions is None:
        allowed_extensions = {'pdf', 'docx'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def save_file(file, user_id):
    if not file or file.filename == '':
        return {"error": "No selected file"}, 400

    if not allowed_file(file.filename):
        return {"error": "File type not allowed. Only PDF and DOCX are supported."}, 400

    user = get_user_by_id(user_id)
    if not user:
        return {"error": "Invalid user ID. User does not exist."}, 400

    filename = secure_filename(file.filename)

    upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')
    os.makedirs(upload_folder, exist_ok=True)

    abs_file_path = os.path.join(upload_folder, filename)
    file.save(abs_file_path)

    rel_path_parts = abs_file_path.split('uploads')
    if len(rel_path_parts) > 1:
        rel_file_path = 'uploads' + rel_path_parts[1]
    else:
        rel_file_path = os.path.join('uploads', filename)
    file_size = os.path.getsize(abs_file_path)

    db = get_db()
    file_record = Resume(user_id=ObjectId(user_id), file_path=rel_file_path)
    db.resumes.insert_one(file_record.to_dict())
    return {
        "message": "File uploaded successfully",
        "filename": filename,
        "size_bytes": file_size,
        "path": rel_file_path
    }, 201
