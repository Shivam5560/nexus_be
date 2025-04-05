
import json
import re
import string
import numpy as np
from typing import Dict, List, Any # For type hinting

# --- Flask and LlamaIndex related imports ---
from flask import Flask, request, jsonify,current_app # Assuming Flask context

from sklearn.metrics.pairwise import cosine_similarity
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Technical keywords list can still be useful for validating/filtering skills if needed
TECHNICAL_KEYWORDS_SEED = set([
    'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'swift', 'kotlin', 'typescript', 'sql', 'nosql', 'scala', 'perl', 'r',
    'react', 'angular', 'vue', 'django', 'flask', 'spring', 'springboot', 'nodejs', 'express', 'rubyonrails', 'laravel', 'jquery', 'bootstrap',
    'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle', 'sqlserver', 'elasticsearch', 'dynamodb',
    'aws', 'azure', 'gcp', 'amazonwebservices', 'googlecloudplatform', 'microsoftazure', 'heroku', 'kubernetes', 'docker', 'terraform', 'lambda', 'ec2', 's3', 'rds',
    'linux', 'unix', 'windows', 'macos', 'bash', 'shell', 'nginx', 'apache',
    'pandas', 'numpy', 'scipy', 'sklearn', 'scikitlearn', 'tensorflow', 'pytorch', 'keras', 'machinelearning', 'deeplearning', 'dataanalysis', 'nlp', 'computervision', 'statistics',
    'git', 'svn', 'jenkins', 'cicd', 'devops', 'agile', 'scrum', 'rest', 'graphql', 'api', 'microservices', 'oop', 'testing', 'debugging', 'jira',
    'architecture', 'scalability', 'performance', 'security', 'algorithms', 'datastructures', 'ux', 'ui',
    'android', 'ios', 'reactnative', 'flutter', 'api', 'sdk', 'ide', 'automation', 'bigdata', 'hadoop', 'spark'
]) # Keep expanding

def get_embed_model():
    embedding_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return embedding_model

embed_model = get_embed_model()

# --- Text Preprocessing (Still useful for dictionary values) ---
def preprocess_text(text: Any) -> str:
    """Cleans and preprocesses text: lowercase, remove punctuation, strip extra whitespace."""
    if not isinstance(text, str): return ""
    text = text.lower()
    # Keep words, spaces, hyphens (common in skills)
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Similarity Calculation (Compares two strings) ---
def calculate_similarity(text1: str, text2: str, embed_model: Any) -> float:
    """Calculates cosine similarity between two preprocessed texts."""
    proc_text1 = preprocess_text(text1)
    proc_text2 = preprocess_text(text2)

    if not proc_text1 or not proc_text2: return 0.0
    try:
        embeddings = embed_model.get_text_embedding_batch([proc_text1, proc_text2], show_progress=False)
        if embeddings is None or len(embeddings) < 2 or embeddings[0] is None or embeddings[1] is None: return 0.0

        embedding1 = np.array(embeddings[0], dtype=np.float32).reshape(1, -1)
        embedding2 = np.array(embeddings[1], dtype=np.float32).reshape(1, -1)

        if not np.isfinite(embedding1).all() or not np.isfinite(embedding2).all(): return 0.0

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0: return 0.0

        embedding1_normalized = embedding1 / norm1
        embedding2_normalized = embedding2 / norm2

        similarity_score = cosine_similarity(embedding1_normalized, embedding2_normalized)[0][0]
        return max(0.0, min(1.0, float(similarity_score)))

    except Exception as e:
        print(f"Error calculating similarity ('{proc_text1[:30]}...' vs '{proc_text2[:30]}...'): {e}")
        return 0.0

# --- Rewritten ATS Logic (Dictionary Comparison) ---
def advanced_ats_similarity(resume_dict: Dict[str, Any], job_description_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates ATS score by comparing structured resume and job description dictionaries.
    """
    embed_model = get_embed_model()
    results = {
        "similarity_score": 0.0, "pass": False, "keywords_missing": [],
        "keywords_found_count": 0, "total_keywords_in_jd": 0,
        "keyword_match_percentage": 0.0, "section_scores": {
            "experience_and_projects": 0.0, "skills": 0.0,
        }, "notes": ""
    }
    #print(job_description_dict)
    # --- 1. Skills Comparison ---
    # Use preprocessed skills for cleaner comparison
    resume_skills_list = [preprocess_text(s) for s in resume_dict.get("skills", []) if isinstance(s, str) and s]
    resume_skills_set = set(resume_skills_list)

    jd_req_skills_list = [preprocess_text(s) for s in job_description_dict.get("required_skills", []) if isinstance(s, str) and s]
    jd_pref_skills_list = [preprocess_text(s) for s in job_description_dict.get("preferred_skills", []) if isinstance(s, str) and s]
    jd_req_skills_set = set(jd_req_skills_list)
    jd_pref_skills_set = set(jd_pref_skills_list)

    # Keyword matching based on required skills
    required_keywords_found = jd_req_skills_set.intersection(resume_skills_set)
    required_keywords_missing = jd_req_skills_set - resume_skills_set
    results["keywords_found_count"] = len(required_keywords_found)
    results["total_keywords_in_jd"] = len(jd_req_skills_set)
    results["keywords_missing"] = sorted(list(required_keywords_missing))
    if results["total_keywords_in_jd"] > 0:
        results["keyword_match_percentage"] = round((results["keywords_found_count"] / results["total_keywords_in_jd"]) * 100, 2)

    # Calculate skills score (weighted overlap)
    req_overlap = len(required_keywords_found) / len(jd_req_skills_set) if jd_req_skills_set else 1.0
    pref_overlap = len(jd_pref_skills_set.intersection(resume_skills_set)) / len(jd_pref_skills_set) if jd_pref_skills_set else 1.0
    overlap_score = 0.7 * req_overlap + 0.3 * pref_overlap
    skills_final_score = overlap_score
    results["section_scores"]["skills"] = round(max(0.0, min(1.0, skills_final_score)) * 100, 2)

    # --- 2. Work Experience and Projects Comparison ---
    # Concatenate relevant text from work experience and projects for semantic comparison
    resume_exp_proj_text = ""
    for exp in resume_dict.get("work_experience", []):
        if isinstance(exp, dict):
            resp_text = "\n".join(exp.get("responsibilities", [])) if isinstance(exp.get("responsibilities"), list) else ""
            resume_exp_proj_text += f"{exp.get('job_title', '')} {resp_text} "

    for proj in resume_dict.get("projects", []):
        if isinstance(proj, dict):
            resume_exp_proj_text += f"{proj.get('name', '')} {proj.get('description', '')} \n"

    jd_resp_text = " ".join(job_description_dict.get("key_responsibilities", [])) if isinstance(job_description_dict.get("key_responsibilities"), list) else ""
    jd_title = job_description_dict.get("job_title", "") if isinstance(job_description_dict.get("job_title"), str) else ""
    jd_full_exp_text = f"{jd_title} {jd_resp_text}"

    experience_projects_semantic_score = calculate_similarity(resume_exp_proj_text, jd_full_exp_text, embed_model)
    # Potential Enhancement: Compare required_experience_years with calculated resume experience
    results["section_scores"]["experience_and_projects"] = round(max(0.0, min(1.0, experience_projects_semantic_score)) * 100, 2)

    # --- 3. Calculate Final Weighted Score ---
    # Determine if job is technical (based on extracted JD dict content)
    jd_skills_combined = jd_req_skills_set.union(jd_pref_skills_set)
    # Use job title from dict and skills overlap with seed list
    jd_title_lower = job_description_dict.get('job_title','').lower() if isinstance(job_description_dict.get('job_title'), str) else ""
    is_technical_job = any(skill in TECHNICAL_KEYWORDS_SEED for skill in jd_skills_combined) or \
                       any(indicator in jd_title_lower for indicator in ['engineer', 'developer', 'programmer', 'scientist', 'technical'])

    if is_technical_job:
        weight_factors = [0.45, 0.55] # Work Exp, Projects, Skills
    else:
        weight_factors = [0.60, 0.40] # Work Exp, Projects, Skills

    # Ensure weights sum to approx 1 (handle potential floating point issues)
    if not abs(sum(weight_factors) - 1.0) < 1e-6:
        print(f"Warning: Weights {weight_factors} do not sum to 1. Normalizing.")
        sum_weights = sum(weight_factors)
        if sum_weights > 0:
            weight_factors = [w / sum_weights for w in weight_factors]
        else: # Handle zero sum case, maybe default to equal weights
            num_factors = len(weight_factors)
            weight_factors = [1.0 / num_factors] * num_factors if num_factors > 0 else []


    final_weighted_score = np.average(
        [results["section_scores"]["experience_and_projects"] / 100.0,
         results["section_scores"]["skills"] / 100.0],
        weights=weight_factors
    )

    # Optional: Add small bonus based on required keyword match percentage
    keyword_bonus_factor = 0.15
    bonus = final_weighted_score * keyword_bonus_factor * (results["keyword_match_percentage"] / 100.0)
    final_score_combined = final_weighted_score + bonus

    score_percent = round(max(0.0, min(1.0, final_score_combined)) * 100, 2)
    pass_threshold = 80.0 # Adjust as needed

    results["similarity_score"] = score_percent
    results["pass"] = score_percent >= pass_threshold
    results["notes"] = f"Dict-Compare Score based on {EMBEDDING_MODEL_NAME}. Pass: {pass_threshold}%. Tech job: {is_technical_job}."

    return results
