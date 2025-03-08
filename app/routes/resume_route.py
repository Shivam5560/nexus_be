from flask import Blueprint, request, jsonify
from app.controllers.resume_controller import upload_file, analyze_resume

resume_bp = Blueprint('resume', __name__, url_prefix="/api/resumes")

@resume_bp.route('/upload', methods=['POST'])
def upload_file_route():
    if not request.files:
        return jsonify({"msg": "Missing file in request"}), 400
    return upload_file()

@resume_bp.route('/analyze', methods=['POST'])
def analyze_resume_route():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400
    return analyze_resume()

def register_resume_routes(app):
    app.register_blueprint(resume_bp)

