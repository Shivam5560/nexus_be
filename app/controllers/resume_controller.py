import os
import json
from flask import request, jsonify, current_app
from openai import embeddings
from werkzeug.utils import secure_filename

from app.services.query_service import generate_query_engine
from app.services.resume_analyzer_service import PracticalResumeAnalyzer
from app.utils.file_util import allowed_file
from app.utils.resume_template import TEMPLATE
from app.utils.text_util import advanced_ats_similarity, get_embed_model

analyzer = PracticalResumeAnalyzer()
embedding_model = get_embed_model()

def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Only PDF and DOCX are supported."}), 400

    # Secure filename and create unique name
    filename = secure_filename(file.filename)

    # Ensure upload folder exists
    upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    file_size = os.path.getsize(file_path)

    return jsonify({
        "message": "File uploaded successfully",
        "filename": filename,
        "size_bytes": file_size,
        "path": file_path
    }), 201


def analyze_resume():
    data = request.json
    if not data or 'file_paths' not in data or 'job_description' not in data:
        return jsonify({"error": "Missing required parameters: file_paths and job_description"}), 400

    file_paths = data['file_paths']
    job_description = data['job_description']

    try:
        query_engine, documents = generate_query_engine(file_paths,embedding_model)
        if not query_engine or not documents:
            return jsonify({"error": "Failed to process documents"}), 500

        resume_str = ""
        for doc in documents:
            resume_str += doc.text_resource.text

        TEMPLATE
        current_template = TEMPLATE.replace("[Insert resume text here]", resume_str)

        response = query_engine.query(current_template).response
        response = response[7:len(response) - 3]

        resume_dict = json.loads(response)
        technical = advanced_ats_similarity(resume_dict, job_description)
        # Analyze a resume
        print(technical)
        grammar_score, recommendations, section_scores = analyzer.analyze_resume(
            resume_str,
            resume_dict,
            industry="tech"
        )

        overall_score = (technical['similarity_score'] * 0.6 + grammar_score * 0.4) * 1.1
        print("Overall_Score", overall_score)
        analysis_results = {
            "resume_data": dict(resume_dict),
            "grammar_analysis": {
                "score": grammar_score,
                "recommendations": recommendations,
                "section_scores": section_scores
            },
            "overall_score": min(round(overall_score, 2), 100)
        }
        return jsonify(analysis_results), 200

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Error decoding resume data: {e}. Ensure your resume format is correct."}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
