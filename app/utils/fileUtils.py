import os
from werkzeug.utils import secure_filename
from flask import current_app

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def save_files(files):
    """
    Save uploaded files to the configured upload folder
    
    Args:
        files: List of file objects from Flask request
        
    Returns:
        list: Paths of saved files
    """
    file_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
    return file_paths