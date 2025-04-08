TEMPLATE = """
This parser leverages a Retrieval-Augmented Generation (RAG) pipeline, where resume text is semantically chunked, indexed in a vector store (e.g., Pinecone), and relevant sections are dynamically retrieved to support precise extraction through LLM inference.
---

### **Instructions**

1.  **Read the Resume Carefully** Analyze the provided resume text thoroughly and extract all relevant details under appropriate subheaders. Preserve the order of sections as they appear in the resume.

2.  **Extract Specific Information** Extract the following details and organize them into the specified structure do not hallucinate or repeat:
    
    * **Personal Information**: Name, email, phone number.
    * **Education**: Degree(s), institution(s), and graduation date(s).
    * **Work Experience/Professional Experience**: Job title, company, employment dates, and key responsibilities (as bullet points or sentences).
    * **Keywords**: Collect all unique technical and professional keywords found throughout the **Work Experience**, **Projects**, and **Skills** sections. Keywords include tools, technologies, programming languages, platforms, libraries, frameworks, databases, software, methodologies, processes, and technical concepts. Do not limit extraction to just the Skills section—also analyze job responsibilities and project descriptions for relevant terms. Avoid duplicates and general words like "team" or "project".
    * **Projects**: Project name and description (multi-line or bullet points).
    * **Certifications**: Certification name and description (if available).
    * **Summary**: A comprehensive professional summary paragraph (minimum 120 words) synthesising the candidate's profile, experience, and key skills, based exclusively on the 'Work Experience' and 'Projects' and 'Skills' sections . This summary must not miss any details of work experience details and projects key descriptions along with skills found within those sections; 

3.  **Structure the Output** Organize the extracted information into the following JSON dictionary format:

```json
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
  "keywords": [],
  "projects": [
    {
      "name": "",
      "description": []
    }
  ],
  "certifications": [
    {
      "name": "",
      "description": []
    }
  ]
  "summary":""
}

4. Fill in the dictionary with the extracted information and in correct order also from the resume by cross-checking with their headers and the extracted value.
5. If any section is not present in the resume, leave it as an empty list or dictionary as appropriate.
6. Ensure all extracted information is accurate and relevant.
7. Return the completed dictionary.
8. Match the dictionary key values with the resume subheaders like personal info and all and do the needful.

Please provide the extracted information in the specified dictionary format. Use JSON format.
"""