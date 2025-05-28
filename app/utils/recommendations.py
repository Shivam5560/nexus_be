from flask import current_app
from llama_index.llms.groq import Groq
import json

def clean_text(s):
    if '```json' in s:
        start_index = s.index('```json')+7
        end_index = s.rindex('```')
    else:
        start_index = 0
        end_index = len(s)
    
    cleaned = s[start_index:end_index]
    return cleaned

def getRecommendations(analysis_dict):
    key = current_app.config.get("GROQ_API_KEY")
    llm = Groq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=key,
        response_format={"type": "json_object"},
        temperature=0.8,
    )
    
    prompt = f"""
    You will be given analysis data in dictionary format. Your task is to produce a concise, professional summary in structured JSON format, divided into two sections: 'refined_recommendations' and 'refined_justifications'.

    Instructions:

    1. **refined_recommendations**:
        - Extract clear, actionable suggestions for improvement.
        - Use concise bullet points (5 points at least).
        - Write in active voice using strong verbs.
        - Avoid filler, fluff, or generic buzzwords.
        - Focus on tangible next steps, not vague advice.

    2. **refined_justifications**:
    - Summarize evaluation or scoring rationale based on content alignment with job criteria.
    - **Do NOT use phrases like "the candidate," "they," or "their."**
    - Use impersonal, objective phrasing like:
        - "Skills cover 5 of 12 job requirements, with strengths in Python but gaps in AWS SageMaker and Apache Beam."
        - "Experience aligns with machine learning model development, data preprocessing, and feature engineering."
    - Use active voice and strong, clear language.
    - Base justifications on facts (e.g., number of skills matched, presence of tools/technologies, scope of experience).
    - Provide **at least 5** meaningful bullet points.
    - Avoid fluff or personal commentary.

    Output Format (JSON only):
    {{
        "refined_recommendations": [
            "Bullet point 1",
            "Bullet point 2",
            ...
        ],
        "refined_justifications": [
            "Bullet point 1",
            "Bullet point 2",
            ...
        ]
    }}

    Input data:
    {analysis_dict}
    """


    
    try:
        response = llm.complete(prompt)
        refined_data = response.text
        refined_data = clean_text(refined_data)
        refined_data = json.loads(refined_data)
        return refined_data
    except Exception as e:
        current_app.logger.error(f"Error generating recommendations: {str(e)}")
        return {
            "refined_recommendations": ["Unable to generate recommendations at this time"],
            "refined_justifications": ["Unable to generate justifications at this time"]
        }