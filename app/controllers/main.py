import tempfile
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load environment variables
load_dotenv()

# Load a pre-trained sentence transformer model
embed_model = HuggingFaceEmbedding(model_name="NeuML/pubmedbert-base-embeddings")
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def calculate_similarity(text1, text2):
    """Enhanced similarity calculation with semantic understanding"""
    embedding1 = embed_model.get_text_embedding(text1)
    embedding2 = embed_model.get_text_embedding(text2)
    similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]
    return (similarity_score + 1) / 2  # Normalize to [0,1]

def extract_keywords(text, num_keywords=15):
    """Extract top keywords using simple tokenization and frequency analysis"""
    tokens = text.split()
    keywords = [word for word in tokens if len(word) > 2]  # Simple filter for keywords
    return sorted(set(keywords), key=lambda x: tokens.count(x), reverse=True)[:num_keywords]

def advanced_ats_similarity(resume_dict, job_description):
    """Enhanced similarity analysis with keyword optimization"""
    job_keywords = extract_keywords(job_description)
    weighted_job_desc = " ".join(job_keywords * 3 + [job_description])

    # Extract semantic embeddings for key sections
    work_exp = " ".join([f"{exp['job_title']} {exp['company']} {' '.join(exp['responsibilities'])}" for exp in resume_dict.get("work_experience", [])])
    projects = " ".join([f"{proj['name']} {proj['description']}" for proj in resume_dict.get("projects", [])])
    skills = " ".join(resume_dict.get("skills", []))

    # Calculate dynamic weighting based on job requirements
    weight_factors = [0.45, 0.25, 0.30] if "technical" in job_description.lower() else [0.35, 0.30, 0.35]

    # Compute similarity scores
    work_score = calculate_similarity(work_exp, weighted_job_desc)
    project_score = calculate_similarity(projects, weighted_job_desc)
    skills_score = calculate_similarity(skills, weighted_job_desc)

    technical_score = np.average([work_score, project_score, skills_score], weights=weight_factors)
    curved_score = min(1.0, technical_score * 1.1)
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

# def analyze_grammar_formatting(resume_text, resume_dict):
#     """Advanced grammar and formatting analysis"""
#     score = 1.0
#     recommendations = []

#     # Advanced Grammar & Spell Check
#     corrected = grammar_checker(resume_text, max_length=1000)[0]['generated_text']
#     # You can implement a simple error count mechanism if needed
#     error_count = sum(1 for word in corrected.split() if word != resume_text.split()[0])  # Simplified error count
#     score -= min(0.3, error_count * 0.02)
#     if error_count > 0:
#         recommendations.append(f"Grammatical issues detected ({error_count} errors). Correct to improve readability.")

#     # Sentiment & Redundancy Analysis
#     blob = TextBlob(resume_text)
#     if blob.sentiment.subjectivity < 0.3:
#         recommendations.append("Increase use of impactful action verbs to highlight achievements.")
#         score -= 0.1

#     # Keyword Alignment Analysis
#     job_keywords = extract_keywords(resume_text)
#     resume_keywords = extract_keywords(resume_text)
#     keyword_overlap = len(set(job_keywords) & set(resume_keywords)) / len(job_keywords) if job_keywords else 0
#     if keyword_overlap < 0.6:
#         score -= 0.15
#         recommendations.append("Improve alignment with job-specific keywords.")

#     # Readability Metrics
#     avg_sentence_length = np.mean([len(sent.split()) for sent in resume_text.split('.') if sent])
#     if avg_sentence_length > 25:
#         score -= 0.1
#         recommendations.append("Simplify sentences; aim for 15-20 words per sentence.")

#     # Passive Voice Check
#     passive_sentences = [sent for sent in resume_text.split('.') if 'by' in sent]  # Simplified passive voice check
#     if len(passive_sentences) > 5:
#         score -= 0.1
#         recommendations.append("Replace passive voice sentences for stronger engagement.")

#     # Action Verb Density
#     action_verbs = ["led", "achieved", "optimized", "designed", "analyzed", "implemented"]
#     work_exp_text = " ".join([exp['job_title'] + ' ' + ' '.join(exp['responsibilities']) for exp in resume_dict.get('work_experience', [])])
#     density = len([w for w in work_exp_text.lower().split() if w in map(str.lower, action_verbs)]) / len(work_exp_text.split()) if work_exp_text else 0
#     if density < 0.05:
#         score -= 0.1
#         recommendations.append("Increase density of strong action verbs in experience description.")

#     score = max(0, min(1.0, score))
#     return round(score * 100, 2), recommendations

def save_files(files):
    """Save uploaded files to upload directory"""
    file_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
    return file_paths

template = """You are an AI assistant trained to extract key information from resumes. Your task is to analyze the given resume text and extract relevant details into a structured dictionary format. Please follow these guidelines:

1. Read the entire resume carefully and extract all the subheaders with all the details in the following format , also do not change the headers subdata the order should be same.
2. Extract the following information:
    * Personal Information (name, email, phone number)
    * Education (degrees, institutions, graduation dates)
    * Work Experience or Professional Experiences (job titles, companies, dates, key responsibilities)
    * Skills
    * Projects (if any)
    * Certifications (if any)
    * Keywords can be technologies, tech keywords or management or soft skills any.
3. Organize the extracted information into a dictionary with the following structure:

{
  "personal_info": [
    {
      "name": "",
      "email": "",
      "phone": ""
    }
  ],
  "education": [
      {
          "degree": "",
          "institution": "",
          "graduation_date": ""
      }
  ],
  "work_experience": [
      {
          "job_title": "",
          "company": "",
          "dates": "",
          "responsibilities": []
      }
  ],
  "skills": [],
  "projects": [
      {
          "name": "",
          "description": ""
      }
  ],
  "certifications": [
          {
          "name": "",
          "description": ""
      }
  ],
  "keywords": []
}

4. Fill in the dictionary with the extracted information and in correct order also from the resume by cross-checking with their headers and the extracted value.
5. If any section is not present in the resume, leave it as an empty list or dictionary as appropriate.
6. Ensure all extracted information is accurate and relevant.
7. Return the completed dictionary.
8. Match the dictionary key values with the resume subheaders like personal info and all and do the needful.

Resume text:
[Insert resume text here]

Please provide the extracted information in the specified dictionary format. Use JSON format.
"""

def generate_query_engine(file_paths):
    """Generates a LlamaIndex query engine from the uploaded PDF files."""
    global template
    try:
        if not file_paths:
            return None, None

        reader = SimpleDirectoryReader(input_files=file_paths)
        documents = reader.load_data()

        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)

        llm = Groq(model="qwen-2.5-32b", api_key=os.getenv("GROQ_API_KEY"), response_format={"type": "json_object"})

        Settings.embed_model = embed_model
        Settings.llm = llm

        vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, node_parser=nodes)
        vector_index.storage_context.persist(persist_dir="./storage_mini")

        storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
        index = load_index_from_storage(storage_context)

        chat_engine = index.as_query_engine(similarity_top_k=2, response_mode="tree_summarize")
        return chat_engine, documents
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

@app.route('/upload', methods=['POST'])
def upload_files():
    """API endpoint to upload resume files"""
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400
    
    try:
        file_paths = save_files(files)
        if not file_paths:
            return jsonify({"error": "No valid files uploaded"}), 400
        
        # Return success message and file paths
        return jsonify({
            "message": "Files uploaded successfully",
            "file_paths": file_paths
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    """API endpoint to analyze resume with job description"""
    data = request.json
    if not data or 'file_paths' not in data or 'job_description' not in data:
        return jsonify({"error": "Missing required parameters: file_paths and job_description"}), 400
    
    file_paths = data['file_paths']
    job_description = data['job_description']
    
    try:
        # Generate query engine from uploaded files
        query_engine, documents = generate_query_engine(file_paths)
        if not query_engine or not documents:
            return jsonify({"error": "Failed to process documents"}), 500
        
        # Extract resume text from documents
        resume_str = ""
        for doc in documents:
            resume_str += doc.text_resource.text
        
        # Replace placeholder in template with actual resume text
        global template
        current_template = template.replace("[Insert resume text here]", resume_str)
        
        # Query the engine with the template to extract resume information
        response = query_engine.query(current_template).response
        response = response[7:len(response)-3]
        # print(response)
        # Parse the JSON response
        resume_dict = json.loads(response)
        # resume_text = ' '.join([str(v) for k, v in resume_dict.items() if isinstance(v, (str, list))])
        # print(resume_dict)
        # Analyze the resume
        technical = advanced_ats_similarity(resume_dict, job_description)
        # grammar_score, recommendations = analyze_grammar_formatting(resume_text, resume_dict)
        
        # Calculate overall score
        overall_score = technical['similarity_score']
        #  + grammar_score * 0.3
        # Prepare response data
        analysis_results = {
            # "resume_data": resume_dict,
            # "technical_analysis": technical,
            "overall_score": min(overall_score,100)
        }
        return jsonify(analysis_results), 200
    
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Error decoding resume data: {e}. Ensure your resume format is correct."}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)