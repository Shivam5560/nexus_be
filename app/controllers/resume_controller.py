import json
from flask import request, jsonify, current_app
import re
from app.services.query_service import generate_query_engine
from app.services.resume_analyzer_service import PracticalResumeAnalyzer
from app.services.file_service import (
    save_file,
    get_resume_by_user_id,
    get_abs_path,
    get_all_resumes_by_user_id,
)
from app.utils.resume_template import TEMPLATE
from app.utils.jd_template import JD_TEMPLATE
from app.utils.text_util import advanced_ats_similarity, get_embed_model
import numpy as np

analyzer = PracticalResumeAnalyzer()
embedding_model = get_embed_model()


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
    logging.info(f"Absolute resume path: {abs_resume_path}")
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
        #print(resume_dict)
        
       

        # 3. Process Job Description String into Dictionary
        job_description_dict = None
        try:
            
            with open('jd.txt','w') as file:
                file.write(job_description)
            file_path2 = 'jd.txt'
            query_engine2, documents2 = generate_query_engine(
            file_path2, embedding_model
            )
            if not query_engine2 or not documents2:
                return jsonify({"error": "Failed to process documents"}), 500

            jd_str = ""
            for doc in documents2:
                jd_str += doc.text_resource.text
          
            jd_prompt = JD_TEMPLATE.replace("[Insert job description text here]", jd_str)
            jd_llm_response_str = query_engine.query(jd_prompt).response
            #with open('test.txt','w') as file:
            #    file.write(str(jd_llm_response_str))
            jd_llm_response_str = jd_llm_response_str[7 : len(jd_llm_response_str) - 3]

            job_description_dict = json.loads(jd_llm_response_str)

        except Exception as e:
            print(f"Error processing job description into dictionary: {e}")
            import traceback
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


        technical = advanced_ats_similarity(resume_dict, job_description_dict)
        # Analyze a resume
        technical = convert_to_normal_types(technical)
        grammar_score, recommendations, section_scores,justifications = analyzer.analyze_resume(
            resume_str, resume_dict, industry="tech"
        )

        overall_score = (
            technical["similarity_score"] * 0.6 + grammar_score * 0.4
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
        #print(analysis_results['grammar_analysis']['score'])
        #print(analysis_results['grammar_analysis']['section_scores'])
        #print(analysis_results['justifications'])
        #print(analysis_results)
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
