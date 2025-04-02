import json
from flask import request, jsonify, current_app

from app.services.query_service import generate_query_engine
from app.services.resume_analyzer_service import PracticalResumeAnalyzer
from app.services.file_service import save_file, get_resume_by_user_id, get_abs_path, get_all_resumes_by_user_id
from app.utils.resume_template import TEMPLATE
from app.utils.text_util import advanced_ats_similarity, get_embed_model

analyzer = PracticalResumeAnalyzer()
embedding_model = get_embed_model()


def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    response, status_code = save_file(file=file,user_id=user_id)
    return jsonify(response), status_code

def get_all_resumes():
    user_id = request.json.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    resumes = get_all_resumes_by_user_id(user_id)
    if not resumes:
        return jsonify({"error": "No resumes found for this user"}), 404
    response_date = {
        "count": len(resumes),
        "list": [resume.to_json() for resume in resumes],
    }
    return jsonify(response_date), 200
    

def analyze_resume():
    data = request.json
    if not data or "user_id" not in data or "job_description" not in data:
        return (
            jsonify(
                {"error": "Missing required parameters: user_id and job_description"}
            ),
            400,
        )

    user_id = data["user_id"]
    job_description = data["job_description"]
    user_data = get_resume_by_user_id(user_id)
    if not user_data:
        return jsonify({"error": "Resume not found for this user"}), 404

    resume_path = user_data.file_path
    abs_resume_path = get_abs_path(resume_path)

    try:
        query_engine, documents = generate_query_engine(
            abs_resume_path, embedding_model
        )
        if not query_engine or not documents:
            return jsonify({"error": "Failed to process documents"}), 500

        resume_str = ""
        for doc in documents:
            resume_str += doc.text_resource.text

        TEMPLATE
        current_template = TEMPLATE.replace("[Insert resume text here]", resume_str)

        response = query_engine.query(current_template).response
        response = response[7 : len(response) - 3]

        resume_dict = json.loads(response)
        technical = advanced_ats_similarity(resume_dict, job_description)
        # Analyze a resume
        print(technical)
        grammar_score, recommendations, section_scores = analyzer.analyze_resume(
            resume_str, resume_dict, industry="tech"
        )

        overall_score = (
            technical["similarity_score"] * 0.6 + grammar_score * 0.4
        ) * 1.1
        print("Overall_Score", overall_score)
        analysis_results = {
            "resume_data": dict(resume_dict),
            "grammar_analysis": {
                "score": grammar_score,
                "recommendations": recommendations,
                "section_scores": section_scores,
            },
            "overall_score": min(round(overall_score, 2), 100),
        }
        return jsonify(analysis_results), 200

    except json.JSONDecodeError as e:
        return (
            jsonify(
                {
                    "error": f"Error decoding resume data: {e}. Ensure your resume format is correct."
                }
            ),
            400,
        )
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
