import json
from flask import request, jsonify, current_app
import re
from app.services.query_service import generate_query_engine
from app.services.resume_analyzer_service import PracticalResumeAnalyzer
from app.services.file_service import (
    save_file,
    # get_resume_by_user_id,
    get_abs_path,
    get_all_resumes_by_user_id,
    get_resume_by_id,
)
from app.utils.resume_template import TEMPLATE
from app.utils.jd_template import JD_TEMPLATE
from app.utils.text_util import advanced_ats_similarity
from app.utils.recommendations import getRecommendations
import numpy as np
import traceback
import random
import string



def clean_text(s):
    if '```json' in s:
        start_index = s.index('```json')+7
        end_index = s.rindex('```')
    else:
        start_index = 0
        end_index = len(s)
    
    cleaned = s[start_index:end_index]
    return cleaned
    


def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    response, status_code = save_file(file=file, user_id=user_id)
    return jsonify(response), status_code


def get_all_resumes(user_id):
    resumes = get_all_resumes_by_user_id(user_id)
    if not resumes:
        return jsonify({"error": "No resumes found for this user"}), 404
    response_data = {
        "count": len(resumes),
        "list": [resume.to_json() for resume in resumes],
    }
    return jsonify(response_data), 200

def analyze_resume():
    analyzer = PracticalResumeAnalyzer()
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
    resume_id = data["resume_id"]
    user_data = get_resume_by_id(user_id,resume_id)
    if not user_data:
        return jsonify({"error": "Resume not found for this user"}), 404

    resume_path = user_data.file_path
    abs_resume_path = get_abs_path(resume_path)
    try:
        query_engine, documents =  generate_query_engine(abs_resume_path,resume_id,read_from_text=False)
        if not query_engine or not documents:
            return jsonify({"error": "Failed to process documents"}), 500

        #print("Resume string started")
        resume_str = ""
        for doc in documents:
            resume_str += doc.text_resource.text
        #print("Resume string closed")
        
        #print("Resume Query Started")
        response =  query_engine.query(TEMPLATE)
        response = response.response
        #print("Resume Query Stopped")
        # with open('res.txt','w') as file:
        #     file.write(str(response))
        response = clean_text(response)
        resume_dict = json.loads(response)
        

        # 3. Process Job Description String into Dictionary
        job_description_dict = None
        jd_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        try:
            query_engine_jd, documents_jd = generate_query_engine(
            job_description,jd_id,read_from_text=True,jd=True
            )
            if not query_engine_jd or not documents_jd:
                return jsonify({"error": "Failed to process documents"}), 500

            response_jd = query_engine_jd.query(JD_TEMPLATE)
            response_jd = response_jd.response
            response_jd = clean_text(response_jd)
            job_description_dict = json.loads(response_jd)

        except Exception as e:
            print(f"Error processing job description into dictionary: {e}")
            traceback.print_exc()
            return jsonify({"error": f"Failed to process job description: {e}"}), 500
        
        def convert_to_normal_types(data):
            """Recursively converts NumPy types within a dictionary to standard Python types."""
            new_data = {}
            for key, value in data.items():
                if isinstance(value, np.generic):
                    new_data[key] = value.item()  # Convert NumPy scalar to Python scalar
                elif isinstance(value, dict):
                    new_data[key] = convert_to_normal_types(value)
                elif isinstance(value, list):
                    new_data[key] = [item.item() if isinstance(item, np.generic) else item for item in value]
                else:
                    new_data[key] = value
            return new_data
        # with open("resume.json", "w") as file:
        #     json.dump(resume_dict, file, indent=4)
        # with open("jd.json", "w") as file:
        #     json.dump(job_description_dict, file, indent=4)
        #print("Starting Technical Analysis")
        technical =  advanced_ats_similarity(resume_dict, job_description_dict)
        # Analyze a resume
        #print("Completed Technical Analysis")
        technical = convert_to_normal_types(technical)
        #print("Starting grammar analysis")
        grammar_score, recommendations, section_scores,justifications = analyzer.analyze_resume(
            resume_str, resume_dict, industry=job_description_dict['industry']
        )
        #print("Completed grammar analysis")
        overall_score = (
            technical["similarity_score"] * 0.55 + grammar_score * 0.45
        )
        print("Overall_Score", overall_score)
        analysis_results = {
            "overall_score": min(round(overall_score, 2), 100),
            "technical_score":technical,
            "grammar_analysis": {
                "score": grammar_score,
                "recommendations": recommendations,
                "section_scores": section_scores,
            },
            "justifications":justifications,
            "resume_data": dict(resume_dict),
        }
        refined_out = getRecommendations(analysis_results)
        analysis_results.update(refined_out)
        return jsonify(analysis_results), 200

    except json.JSONDecodeError as e:
        traceback.print_exc()
        return (
            jsonify(
                {
                    "error": f"Error decoding resume data: {e}. Ensure your resume format is correct."
                }
            ),
            400,
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

