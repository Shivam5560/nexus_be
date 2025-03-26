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
    os.makedirs(current_app.config.get('UPLOAD_FOLDER'), exist_ok=True)
    file_path = os.path.join(current_app.config.get('UPLOAD_FOLDER'), filename)
    file.save(file_path)

    db = get_db()
    file_record = Resume(user_id=ObjectId(user._id),file_path=file_path)
    db.resumes.insert_one(file_record.to_dict())
    file_size = os.path.getsize(file_path)

    return {
        "message": "File uploaded successfully",
        "filename": filename,
        "size_bytes": file_size,
        "path": file_path
    }, 201

def get_abs_path(relative_path):
    upload_folder = current_app.config.get("UPLOAD_FOLDER")
    upload_folder = os.path.abspath(upload_folder)
    absolute_path = os.path.join(upload_folder, os.path.basename(relative_path))

    return absolute_path

def get_resume_by_user_id(user_id):
    db = get_db()
    resume = db.resumes.find_one({"user_id": ObjectId(user_id)})
    if resume:
        return Resume.from_dict(resume)
    return None

def get_all_resume_by_user_id(user_id):
    db = get_db()
    resumes = db.resumes.find_all({"user_id":ObjectId(user_id)})
    if resumes:
        return Resume.from_dict(resumes)
    return None