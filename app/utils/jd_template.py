JD_TEMPLATE = """
This parser is designed to extract structured information from a plain job description text using LLM inference only — no retrieval or external database involved.
---

### **Instructions**

1. **Read the Job Description Carefully**  
   Analyze the provided job description content thoroughly and extract all relevant details using accurate phrasing and concise keywords.

2. **Extract the Following Information**  
   Use only the content provided in the job description. Do not infer, guess, or hallucinate information. If a field is not explicitly stated, return an empty string `""` or an empty list `[]` as appropriate.

    * **Job Title**: The main title of the role (e.g., "Data Analyst").
    * **Company Name**: Name of the company if mentioned.
    * **Location**: Location of the job (e.g., "Remote", "Berlin", "Hybrid - NYC").
    * **Required Skills**: A cleaned-up list of technical tools, concepts, soft skills, and technologies. Convert phrases like "Python skills" → "Python", and expand acronyms like "ML" → "Machine Learning".
    * **Required Experience (Years)**: Minimum number of years if explicitly mentioned (e.g., 3, 5).
    * **Key Responsibilities**: A detailed paragraph (~500 tokens) that provides a thorough overview of what this role entails, including key deliverables, responsibilities, and the skills needed to perform them effectively.
    * **Other Qualifications**: Extra credentials or requirements, such as degrees, certifications, or industry-specific experience (e.g., "B.S. in Computer Science", "PMP Certification").

3. **Structure the Output**  
   Use the following JSON format:

```json
{
  "job_title": "",
  "company_name": "",
  "location": "",
  "required_skills": [],
  "required_experience_years": "",
  "key_responsibilities": "",
  "other_qualifications": []
}
"""