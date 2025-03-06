import os
from flask import request, jsonify
from app.utils.fileUtils import saveFile

def uploadFile():
    file = request.files.get(file)

    if not file or file.filename == '':
        return jsonify({"msg": "No file uploaded"}), 200
    
    file_path, error = saveFile(file)
    if error:
        return jsonify({"msg":error}),400
    
    return jsonify({"msg": "File uploaded successfully" , "path": file_path}),200