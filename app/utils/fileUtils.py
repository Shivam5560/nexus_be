import os
from flask import request , current_app;
from werkzeug.utils import secure_filename;


def allowedFile(filename):
    allowed_extension = current_app.config['ALLOWED_EXTENSIONS']
    return '.' in filename and filename.rsplit('.',1)[1].lower in allowed_extension

def saveFile(file):
    if not allowedFile(file.filename):
        return None, "Invalid file type"
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
    file.save(file_path)
    return file_path, None 