JD_TEMPLATE = """
**Objective:** Analyze the provided job description text. Extract key information precisely and structure it as a single, valid JSON object according to the schema below. Focus on identifying specific skills (both technical and soft) as keywords.

**Input Job Description:**
[Insert job description text here]

**Instructions:**

1.  **Parse Thoroughly:** Read the entire job description carefully.
2.  **Identify Keywords:** Extract specific, concise keywords for skills (technical), tools, technologies, and qualifications. 
3.  **Adhere to Schema:** Populate the JSON object strictly following the specified fields and data types. Use `null` or empty lists (`[]`) where information is not found or not applicable.
4.  **Generate Single JSON:** Ensure the final output is one complete, valid JSON object.

**Desired JSON Schema:**

```json
{
  "job_title": "string", // The primary job title mentioned (e.g., "Software Engineer", "Data Analyst").
  "company_name": "string_or_null", // The name of the hiring company, or null if not specified.
  "location": "string_or_null", // The primary work location (e.g., "Remote", "New York, NY", "Hybrid - London"), or null if not specified.
  "required_skills": [
    // Extract a list of keywords representing skills, technologies, tools, methodologies, domains, qualifications, or concepts explicitly stated as REQUIRED or ESSENTIAL for the role.
    // Focus  on items identified as mandatory requirements,  any preferred, 'nice-to-have', or 'bonus' skills.
    // Examples can include technical skills ("Python", "SQL", "AWS"), tools ("Jira", "Docker"), concepts ("Machine Learning", "Data Structures"), languages ("English"), or soft skills ("Communication", "Analytical Thinking").
    // Clean the extracted keywords by removing generic trailing words like "skills", "ability", "knowledge", "proficiency", "certification", "degree", "experience".
    // For example: "Communication skills" becomes "Communication", "Proficiency in SQL" becomes "SQL", "Experience with AWS" becomes "AWS", "Agile certification" becomes "Agile".
    // Extract terms/keywords as accurately as possible from the text.
    // Expand common short technical terms to their full names (e.g., "ML" to "Machine Learning", "NLP" to "Natural Language Processing").
    // Return a list of unique strings. Return an empty list [] if no required skills/qualifications are explicitly listed.
    // Example Output: ["Python", "SQL", "Amazon Web Services", "Machine Learning", "Communication", "Agile"]
    "string"
  ]
  "required_experience_years": "integer_or_null", // Minimum years of *relevant* experience explicitly required (e.g., if text says "5+ years" or "minimum 5 years", use 5). Use null if no specific minimum duration is mentioned.
"key_responsibilities":
    // Generate ONE single paragraph summarizing the core duties, tasks, and responsibilities for this role based on the job description.
    // Integrate key details and technical keywords mentioned throughout the JD, ensuring alignment with the required skills.
    // Weave in skills naturally within the summary, for example: "...involves deploying models using Docker and AWS SageMaker" or "requires proficiency in Python, SQL, and machine learning libraries like TensorFlow/PyTorch."
    // The paragraph should provide a cohesive overview of the role's main functions and expectations.
    // Aim for a reasonably detailed summary, ideally around 100 words or more, capturing the essence of the role.
    // If no specific responsibilities are listed or inferrable from the job description, return an empty string "".
    // DO NOT list responsibilities as bullet points or separate short strings; create a unified descriptive paragraph.
    "string"
,
  "other_qualifications": [
     // List of other beneficial qualifications, experiences, or attributes not categorized as skills (required or preferred).
     // Includes: Certifications (e.g., "PMP Certification", "AWS Certified Developer"), specific industry experience (e.g., "FinTech experience", "Healthcare background"), educational degrees (if mentioned as beneficial beyond a baseline requirement), security clearance, etc.
     // Example: ["B.S. in Computer Science", "Experience in Agile environments", "Security Clearance Eligible"]
     // Empty list [] if none apply.
    "string"
  ]
}"""
