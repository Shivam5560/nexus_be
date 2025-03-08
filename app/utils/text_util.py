from flask import current_app
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def get_embed_model():
    embedding_model = HuggingFaceEmbedding(model_name="neuml/pubmedbert-base-embeddings")
    return embedding_model

embed_model = get_embed_model()

def calculate_similarity(text1, text2):
    embedding1 = embed_model.get_text_embedding(text1)
    embedding2 = embed_model.get_text_embedding(text2)
    similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]
    return (similarity_score + 1) / 2

def extract_keywords(text, num_keywords=15):
    tokens = text.split()
    keywords = [word for word in tokens if len(word) > 2]
    return sorted(set(keywords), key=lambda x: tokens.count(x), reverse=True)[:num_keywords]

def advanced_ats_similarity(resume_dict, job_description):
    job_keywords = extract_keywords(job_description)
    weighted_job_desc = " ".join(job_keywords * 3 + [job_description])

    work_exp = " ".join([f"{exp['job_title']} {exp['company']} {' '.join(exp['responsibilities'])}" for exp in
                         resume_dict.get("work_experience", [])])
    projects = " ".join([f"{proj['name']} {proj['description']}" for proj in resume_dict.get("projects", [])])
    skills = " ".join(resume_dict.get("skills", []))

    weight_factors = [0.45, 0.25, 0.30] if "technical" in job_description.lower() else [0.35, 0.30, 0.35]

    work_score = calculate_similarity(work_exp, weighted_job_desc)
    project_score = calculate_similarity(projects, weighted_job_desc)
    skills_score = calculate_similarity(skills, weighted_job_desc)

    technical_score = np.average([work_score, project_score, skills_score], weights=weight_factors)
    curved_score = min(1.0, technical_score * 1.2)
    score = round(curved_score * 100, 2)

    return {
        "similarity_score": score,
        "pass": score >= 75,
        "keywords_missing": len(set(job_keywords) - set(extract_keywords(work_exp + projects + skills))),
        "section_scores": {
            "work_experience": round(work_score * 100, 2),
            "projects": round(project_score * 100, 2),
            "skills": round(skills_score * 100, 2),
        }
    }


def simple_tokenize_sentences(text):
    """Simple sentence tokenizer without NLTK."""
    # Handle common abbreviations to avoid splitting them
    text = re.sub(r'(\w\.\w\.)', r'\1TEMP', text)
    text = re.sub(r'(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)', r'\1TEMP', text)

    # Split on sentence boundaries
    sentences = re.split(r'(?<!\w\.\w\.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

    # Restore abbreviations
    sentences = [re.sub(r'TEMP', '', s) for s in sentences]

    # Remove empty sentences
    return [s.strip() for s in sentences if s.strip()]


def simple_word_tokenize(text):
    """Simple word tokenizer without NLTK."""
    # Replace punctuation with spaces, then split on whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    return [word for word in text.split() if word]