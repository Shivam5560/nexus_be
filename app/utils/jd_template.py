JD_TEMPLATE = """
This parser leverages a Retrieval-Augmented Generation (RAG) pipeline, where job descriptions are chunked, stored in a vector database, and retrieved dynamically to guide structured extraction using LLM inference.
---

### **Instructions**

1. **Read the Job Description Carefully** Analyze the retrieved job description content thoroughly and extract all relevant details using accurate phrasing and concise keywords.

2. **Extract the Following Information** Use only the provided content, do not guess or hallucinate. Return missing values as `null` or `[]` as appropriate.

    * **Job Title**: The main title of the role (e.g., "Data Analyst").
    * **Company Name**: Name of the company if mentioned.
    * **Location**: Location of the job (e.g., "Remote", "Berlin", "Hybrid - NYC").
    * **Required Skills**: A cleaned-up list of technical tools, concepts, soft skills, and technologies. Convert phrases like "Python skills" → "Python", and expand acronyms like "ML" → "Machine Learning".
    * **Required Experience (Years)**: Minimum number of years if explicitly mentioned (e.g., 3, 5).
    * **Key Responsibilities**: A detailed paragraph (~500 tokens) summarizing the major responsibilities and required capabilities. Include technical and soft skills naturally in the summary.
    * **Other Qualifications**: Extra credentials or requirements, such as degrees, certifications, or industry-specific experience (e.g., "B.S. in Computer Science", "PMP Certification").

3. **Structure the Output** Use the following JSON format:

```json
{
  "job_title": "string",
  "company_name": "string_or_null",
  "location": "string_or_null",
  "required_skills": ["string"],
  "required_experience_years": "integer_or_null",
  "key_responsibilities": "string",
  "other_qualifications": ["string"]
}
"""