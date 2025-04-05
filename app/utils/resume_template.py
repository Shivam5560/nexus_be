TEMPLATE = """You are an AI assistant trained to extract key information from resumes. Your task is to analyze the given resume text and extract relevant details into a structured dictionary format. Please follow these guidelines:

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
