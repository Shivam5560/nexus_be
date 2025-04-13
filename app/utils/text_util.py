import json
import re
import string
import numpy as np
from typing import Dict, List, Any, Tuple # Added Tuple

# --- Flask and LlamaIndex related imports ---
# Assuming these are correctly set up in your environment
# from flask import Flask, request, jsonify, current_app
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.embeddings.cohere import CohereEmbedding
from flask import current_app


TECHNICAL_KEYWORDS_SEED = set([
    'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'swift', 'kotlin', 'typescript', 'sql', 'nosql', 'scala', 'perl', 'r',
    'react', 'angular', 'vue', 'django', 'flask', 'spring', 'springboot', 'nodejs', 'express', 'rubyonrails', 'laravel', 'jquery', 'bootstrap',
    'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle', 'sqlserver', 'elasticsearch', 'dynamodb',
    'aws', 'azure', 'gcp', 'amazonwebservices', 'googlecloudplatform', 'microsoftazure', 'heroku', 'kubernetes', 'docker', 'terraform', 'lambda', 'ec2', 's3', 'rds',
    'linux', 'unix', 'windows', 'macos', 'bash', 'shell', 'nginx', 'apache',
    'pandas', 'numpy', 'scipy', 'sklearn', 'scikitlearn', 'tensorflow', 'pytorch', 'keras', 'machinelearning', 'deeplearning', 'dataanalysis', 'nlp', 'computervision', 'statistics',
    'git', 'svn', 'jenkins', 'cicd', 'ci/cd', 'devops', 'agile', 'scrum', 'rest', 'graphql', 'api', 'microservices', 'oop', 'testing', 'debugging', 'jira',
    'architecture', 'scalability', 'performance', 'security', 'algorithms', 'datastructures', 'ux', 'ui',
    'android', 'ios', 'reactnative', 'flutter', 'api', 'sdk', 'ide', 'automation', 'bigdata', 'hadoop', 'spark'
]) # Keep expanding - Added ci/cd

# Define a threshold for semantic skill matching
SKILL_SIMILARITY_THRESHOLD = 0.75 # Adjust this value based on testing (0.80 is a reasonable starting point)

def get_embed_model():
    cohere_api_key = current_app.config.get("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY not found in configuration.")
    # Use 'search_query' for JD skills and 'search_document' for resume skills if asymmetric,
    # or 'search_document' for both if symmetric comparison is desired. Let's use symmetric for now.
    embed_model = CohereEmbedding(
        api_key=cohere_api_key,
        model_name="embed-english-v3.0",
        input_type="search_document", # or "search_query" for JD skills if preferred
    )
    return embed_model

def preprocess_text(text: Any) -> str:
    """Cleans and preprocesses text: lowercase, remove punctuation (keep essential like '/'), strip extra whitespace."""
    if not isinstance(text, str): return ""
    text = text.lower()
    # Keep words, spaces, hyphens, slashes (common in skills like ci/cd)
    text = re.sub(r'[^\w\s\-/]', '', text) # Allow hyphen and slash
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_batch_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Calculates cosine similarity between two batches of embeddings."""
    # Ensure embeddings are 2D arrays
    if embeddings1.ndim == 1: embeddings1 = embeddings1.reshape(1, -1)
    if embeddings2.ndim == 1: embeddings2 = embeddings2.reshape(1, -1)

    # Normalize embeddings
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Handle zero vectors to avoid division by zero
    normalized1 = np.divide(embeddings1, norm1, out=np.zeros_like(embeddings1), where=norm1!=0)
    normalized2 = np.divide(embeddings2, norm2, out=np.zeros_like(embeddings2), where=norm2!=0)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(normalized1, normalized2)
    return np.clip(similarity_matrix, 0.0, 1.0) # Ensure values are between 0 and 1


def find_skill_matches_with_embeddings(
    resume_skills: List[str],
    jd_skills: List[str],
    embed_model: Any,
    threshold: float
) -> Tuple[List[str], List[str], Dict[str, Tuple[str, float]]]:
    """
    Finds semantic matches between JD skills and Resume skills using embeddings.

    Args:
        resume_skills: List of preprocessed skills from the resume.
        jd_skills: List of preprocessed skills from the job description.
        embed_model: The embedding model instance.
        threshold: The minimum cosine similarity score to consider a match.

    Returns:
        A tuple containing:
        - list of found JD skills.
        - list of missing JD skills.
        - dictionary mapping found JD skills to their best matching resume skill and the similarity score.
    """
    if not jd_skills:
        return [], [], {}
    if not resume_skills:
        return [], sorted(list(set(jd_skills))), {} # All JD skills are missing

    unique_resume_skills = sorted(list(set(resume_skills)))
    unique_jd_skills = sorted(list(set(jd_skills)))

    # Embed all unique skills together for efficiency
    all_unique_skills = unique_resume_skills + unique_jd_skills
    try:
        all_embeddings_list = embed_model.get_text_embedding_batch(all_unique_skills, show_progress=False)
        if all_embeddings_list is None or len(all_embeddings_list) != len(all_unique_skills):
             print("Warning: Embedding failed or returned unexpected number of results.")
             # Fallback to no matches if embedding fails
             return [], unique_jd_skills, {}
        all_embeddings = np.array(all_embeddings_list, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embeddings for skills: {e}")
        # Fallback to no matches if embedding fails
        return [], unique_jd_skills, {}

    # Separate embeddings back
    num_resume_skills = len(unique_resume_skills)
    resume_embeddings = all_embeddings[:num_resume_skills]
    jd_embeddings = all_embeddings[num_resume_skills:]

    if resume_embeddings.shape[0] == 0 or jd_embeddings.shape[0] == 0:
         # Handle cases where one list was empty after unique filter, though initial checks should catch this
         return [], unique_jd_skills, {}


    # Calculate similarity matrix: rows=jd_skills, cols=resume_skills
    similarity_matrix = calculate_batch_similarity(jd_embeddings, resume_embeddings) # Shape: (len(unique_jd_skills), len(unique_resume_skills))

    found_skills = []
    missing_skills = []
    match_details = {} # jd_skill -> (best_resume_match, score)

    for i, jd_skill in enumerate(unique_jd_skills):
        best_match_score = np.max(similarity_matrix[i, :]) if similarity_matrix.shape[1] > 0 else 0.0
        best_match_index = np.argmax(similarity_matrix[i, :]) if similarity_matrix.shape[1] > 0 else -1

        if best_match_score >= threshold and best_match_index != -1:
            found_skills.append(jd_skill)
            best_matching_resume_skill = unique_resume_skills[best_match_index]
            match_details[jd_skill] = (best_matching_resume_skill, round(float(best_match_score), 4))
        else:
            missing_skills.append(jd_skill)

    return sorted(found_skills), sorted(missing_skills), match_details


# --- Rewritten ATS Logic (Dictionary Comparison with Semantic Skills) ---
def advanced_ats_similarity(resume_dict: Dict[str, Any], job_description_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates ATS score using semantic similarity for skills and GRANULAR experience/projects.
    """
    try:
        embed_model = get_embed_model()
    except ValueError as e:
         return {"error": str(e), "similarity_score": 0.0, "pass": False}

    results = {
        "similarity_score": 0.0,
        "pass": False,
        "required_skills_missing": [],
        "required_skills_found": [],
        "preferred_skills_found": [],
        "required_skills_found_count": 0,
        "total_required_skills_in_jd": 0,
        "required_skill_match_percentage": 0.0,
        "skill_match_details": {},
        "section_scores": {
            "experience_and_projects": 0.0,
            "skills": 0.0,
        },
        "justification": {
            "skills": "",
            "experience_and_projects": "",
            "overall": "",
        },
        "notes": "",
        "job_description_responsibilities": "", # Add the extracted JD responsibilities
    }

    # --- 1. Skills Comparison (Semantic - Unchanged) ---
    skills_justification_parts = []
    resume_skills_list = [preprocess_text(s) for s in resume_dict.get("keywords", []) if isinstance(s, str) and s]
    skills_justification_parts.append(f"Resume skills processed: {len(set(resume_skills_list))}")
    jd_req_skills_list = [preprocess_text(s) for s in job_description_dict.get("required_skills", []) if isinstance(s, str) and s]
    unique_jd_req_skills = sorted(list(set(jd_req_skills_list)))
    results["total_required_skills_in_jd"] = len(unique_jd_req_skills)


    req_found, req_missing, req_match_details = find_skill_matches_with_embeddings(
        resume_skills_list, unique_jd_req_skills, embed_model, SKILL_SIMILARITY_THRESHOLD
    )

    results["required_skills_found"] = req_found
    results["required_skills_missing"] = req_missing


    results["required_skills_found_count"] = len(req_found)
    results["skill_match_details"] = req_match_details
    skills_justification_parts.append(f"Required skills found ({len(req_found)}/{results['total_required_skills_in_jd']}) using similarity >={SKILL_SIMILARITY_THRESHOLD}: {req_found if req_found else 'None'}.")
    skills_justification_parts.append(f"Required skills missing: {req_missing if req_missing else 'None'}.")
    
    if results["total_required_skills_in_jd"] > 0:
        results["required_skill_match_percentage"] = round((results["required_skills_found_count"] / results["total_required_skills_in_jd"]) * 100, 2)
        skills_justification_parts.append(f"Required skill match percentage: {results['required_skill_match_percentage']}%.")
    else:
        results["required_skill_match_percentage"] = 100.0
        skills_justification_parts.append("No required skills specified in JD.")

    req_overlap = len(req_found) / len(unique_jd_req_skills) if unique_jd_req_skills else 1.0
    skills_final_score = 1 * req_overlap
    results["section_scores"]["skills"] = round(max(0.0, min(1.0, skills_final_score)) * 100, 2)
    skills_justification_parts.append(f"Final Skills Score : {results['section_scores']['skills']}%.")
    results["justification"]["skills"] = " ".join(skills_justification_parts)


    # --- 2. Work Experience and Projects Comparison (SIMPLIFIED using Resume Summary) ---
    exp_proj_justification_parts = []
    similarity_score = 0.0 # Renamed from max_similarity_score for clarity

    # 2a. Prepare Augmented JD Text (Unchanged)
    jd_title = str(job_description_dict.get("job_title", ""))
    jd_resp_list = job_description_dict.get("key_responsibilities", [])
    jd_resp_text = "\n".join(jd_resp_list) if isinstance(jd_resp_list, list) else str(jd_resp_list)
    jd_quals_list = job_description_dict.get("qualifications", [])
    jd_quals_text = "\n".join(jd_quals_list) if isinstance(jd_quals_list, list) else str(jd_quals_list)
    jd_req_skills_text = ", ".join(unique_jd_req_skills) # From skills analysis

    jd_full_exp_text = f"Job Title: {jd_title}\nResponsibilities:\n{jd_resp_text}\nQualifications:\n{jd_quals_text}\nRequired Skills: {jd_req_skills_text}"
    preprocessed_jd_text = preprocess_text(jd_full_exp_text)

    # 2b. Prepare Resume Summary Text (NEW APPROACH)
    resume_summary = str(resume_dict.get("summary", "")) # Get the summary field
    preprocessed_resume_summary = preprocess_text(resume_summary)
    exp_proj_justification_parts.append("Comparing Job Description against the overall Resume Summary.")

    # 2c. Perform Comparison (JD vs Resume Summary)
    if not preprocessed_jd_text:
        exp_proj_justification_parts.append("JD text is empty, cannot perform summary comparison.")
    elif not preprocessed_resume_summary:
        exp_proj_justification_parts.append("Resume summary field is empty or could not be processed.")
    else:
        try:
            # Embed JD Text (as 'search_query')
            original_input_type = embed_model.input_type
            embed_model.input_type = "search_query"
            jd_embedding_list = embed_model.get_text_embedding_batch([preprocessed_jd_text], show_progress=False)
            embed_model.input_type = original_input_type # Change back immediately

            if not jd_embedding_list or jd_embedding_list[0] is None:
                raise ValueError("Failed to embed job description text.")
            jd_embedding = np.array(jd_embedding_list[0], dtype=np.float32).reshape(1, -1)

            # Embed Resume Summary (as 'search_document')
            embed_model.input_type = "search_document"
            # Use batch for consistency, expecting a list of one embedding
            resume_summary_embedding_list = embed_model.get_text_embedding_batch([preprocessed_resume_summary], show_progress=False)
            embed_model.input_type = original_input_type # Change back

            if resume_summary_embedding_list and resume_summary_embedding_list[0] is not None:
                resume_summary_embedding = np.array(resume_summary_embedding_list[0], dtype=np.float32).reshape(1, -1)

                # Calculate Similarity (JD vs Resume Summary)
                similarity_matrix = calculate_batch_similarity(jd_embedding, resume_summary_embedding) # Shape: (1, 1)

                if similarity_matrix.size > 0:
                    similarity_score = float(similarity_matrix[0, 0]) # Direct score
                    score_details = f"Similarity score between JD and resume summary: {similarity_score:.3f}"
                    exp_proj_justification_parts.append(score_details)
                else:
                    # Should not happen if embeddings were generated, but handle defensively
                    similarity_score = 0.0
                    exp_proj_justification_parts.append("Warning: Similarity calculation returned empty.")

            else:
                exp_proj_justification_parts.append("Warning: Failed to embed resume summary text.")
                similarity_score = 0.0

        except Exception as e:
            print(f"Error during summary comparison: {e}")
            similarity_score = 0.0
            exp_proj_justification_parts.append(f"An error occurred during summary comparison: {e}")

    # 2d. Assign Score and Justification
    # Use the single similarity_score calculated above
    results["section_scores"]["experience_and_projects"] = round(max(0.0, min(1.0, similarity_score)) * 100, 2)
    exp_proj_justification_parts.insert(0, f"Calculated as semantic similarity between the Job Description text and the provided Resume Summary.") # Add context
    results["justification"]["experience_and_projects"] = " ".join(exp_proj_justification_parts)


    # --- 3. Calculate Final Weighted Score (Logic Unchanged, uses the new score from Section 2) ---
    overall_justification_parts = []
    jd_skills_combined_set = set(unique_jd_req_skills)
    jd_title_lower = job_description_dict.get('job_title','')
    print("Jd_tile_lower prev = ",jd_title_lower)
    if jd_title_lower!='' or jd_title_lower!='None' or jd_title_lower is not None or jd_title_lower!='null':
        jd_title_lower = jd_title_lower.lower()
    print("Jd_tile_lower after = ",jd_title_lower)

    # Determine if technical based on keywords in JD title or combined skills
    is_technical_job = any(skill in TECHNICAL_KEYWORDS_SEED for skill in jd_skills_combined_set) or \
                    any(indicator in jd_title_lower for indicator in ['engineer', 'developer', 'programmer', 'scientist', 'technical', 'analyst', 'architect', 'data', 'software'])
    overall_justification_parts.append(f"Job classified as technical: {is_technical_job}.")

    if is_technical_job:
        # Weights: Experience/Projects (from Summary comparison), Skills
        weight_factors = [0.30, 0.70]
        overall_justification_parts.append("Using technical weights: Summary/Experience=30%, Skills=70%.")
    else:
        # Weights: Experience/Projects (from Summary comparison), Skills
        weight_factors = [0.40, 0.60]
        overall_justification_parts.append("Using non-technical weights: Summary/Experience=40%, Skills=60%.")

    # Get the scores (already calculated)
    exp_proj_score_norm = results["section_scores"]["experience_and_projects"] / 100.0
    skills_score_norm = results["section_scores"]["skills"] / 100.0

    # Ensure weights sum to 1 (simple normalization)
    total_weight = sum(weight_factors)
    if total_weight > 0 and not np.isclose(total_weight, 1.0):
        weight_factors = [w / total_weight for w in weight_factors]

    # Calculate weighted average
    final_weighted_score = np.average(
        [exp_proj_score_norm, skills_score_norm],
        weights=weight_factors
    ) if len(weight_factors) == 2 else (exp_proj_score_norm + skills_score_norm) / 2 # Fallback average
    overall_justification_parts.append(f"Weighted average score before bonus: {round(final_weighted_score * 100, 2)}%.")

    # Apply bonus based on required skills match percentage
    keyword_bonus_factor = 0.15
    keyword_match_norm = results.get("required_skill_match_percentage", 0) / 100.0 # Use .get for safety
    bonus = final_weighted_score * keyword_bonus_factor * keyword_match_norm
    final_score_combined = final_weighted_score + bonus
    overall_justification_parts.append(f"Required keyword match bonus ({keyword_bonus_factor*100}% factor applied to weighted score * required match %): +{round(bonus * 100, 2)}%.")

    # Final Score Calculation and Pass/Fail
    score_percent = round(max(0.0, min(1.0, final_score_combined)) * 100, 2)
    pass_threshold = 80.0
    results["similarity_score"] = score_percent
    results["pass"] = score_percent >= pass_threshold
    results["notes"] = f"Pass Threshold: {pass_threshold}%. Tech job: {is_technical_job}."
    overall_justification_parts.append(f"Final Score (Weighted Avg + Bonus, capped at 100%): {results['similarity_score']}%. Pass: {results['pass']}.")
    results["justification"]["overall"] = " ".join(overall_justification_parts)

    # Add JD responsibilities to results (as requested in original snippet)
    results["job_description_responsibilities"] = job_description_dict.get("key_responsibilities", []) # Use empty list as default

    # (Assuming the function returns results)
    return results