TEMPLATE = """
**Prompt: Resume Information Extraction**

You are an AI assistant trained to extract key information from resumes and organize it into a structured dictionary format. Follow these instructions carefully:

---

### **Instructions**

1.  **Read the Resume Carefully** Analyze the provided resume text thoroughly and extract all relevant details under appropriate subheaders. Preserve the order of sections as they appear in the resume.

2.  **Extract Specific Information** Extract the following details and organize them into the specified structure:
    
    * **Personal Information**: Name, email, phone number.
    * **Education**: Degree(s), institution(s), and graduation date(s).
    * **Work Experience/Professional Experience**: Job title, company, employment dates, and key responsibilities (as bullet points or sentences).
    * **Keywords**: Consolidate all identified keywords representing skills, technologies, tools, languages, methodologies, concepts, etc., found anywhere into the 'keywords' list as per its detailed instructions in the dictionary structure below. (This replaces the 'Skills' bullet point).
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
  "keywords": [
    // CRUCIAL: Extract a comprehensive list of ALL keywords representing skills, technologies, tools, platforms, frameworks, libraries, methodologies, concepts, languages, or domains mentioned ANYWHERE in the resume text.
    // **Actively scan the 'Work Experience' responsibilities and 'Projects' descriptions/technologies sections.** It is vital to capture keywords demonstrating practical application (e.g., "developed API using Python/Flask", "managed AWS EC2 instances", "analyzed data with Pandas", "led Agile sprints"). Extract these mentioned technologies, tools, platforms, methodologies, libraries, and technical concepts and add them to this keywords list, even if they are not listed in a dedicated 'Skills' section. Do NOT limit your search to only sections explicitly titled 'Skills'.
    // Examples to capture broadly include: technical items ("Python", "Java", "AWS", "React", "Docker", "Git", "SQL", "Pandas"), software ("Salesforce", "Excel", "Jira"), concepts ("Machine Learning", "Data Analysis", "OOP", "REST API"), methodologies ("Agile", "Scrum", "CI/CD"), soft skills ("Leadership", "Teamwork", "Problem Solving") etc., wherever they are mentioned.
    // Consolidate ALL identified keywords (representing skills, technologies, tools, etc.) from all sections into this single list. DO NOT create separate keys for different types (e.g., 'Technical Skills', 'Soft Skills').
    // Clean the extracted keywords by removing generic trailing words like "skills", "ability", "knowledge", "proficiency", "expertise", "language", "framework", "tools", "concepts".
    // For example: "Communication skills" becomes "Communication", "Proficiency in SQL" becomes "SQL", "Expertise with Java" becomes "Java", "Knowledge of Agile" becomes "Agile".
    // Extract terms/keywords accurately as they appear in the resume. Avoid expanding abbreviations unless the full form is clearly provided nearby (e.g., "Machine Learning (ML)"). If both short and full forms appear, prefer the most complete form mentioned.
    // Return a list of unique strings (case-insensitive matching for uniqueness can be helpful before finalizing). Return an empty list [] if no relevant keywords are identified in the resume.
    "List[string]" // Expecting a list of strings
   ],
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

Resume text:
[Insert resume text here]

Please provide the extracted information in the specified dictionary format. Use JSON format.
"""