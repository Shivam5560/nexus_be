JD_TEMPLATE = """
**Objective:** Analyze the provided job description text. Extract key information precisely and structure it as a single, valid JSON object according to the schema below. Focus on identifying specific skills (both technical and soft) as keywords.

**Input Job Description:**
[Insert job description text here]

**Instructions:**

1.  **Parse Thoroughly:** Read the entire job description carefully.
2.  **Identify Keywords:** Extract specific, concise keywords for skills (technical), tools, technologies, and qualifications. Avoid full sentences or descriptions in skill lists.
3.  **Categorize Explicitly:** Differentiate between requirements stated as mandatory ("required", "must have", "essential") and preferences ("preferred", "plus", "nice to have", "bonus").
4.  **Adhere to Schema:** Populate the JSON object strictly following the specified fields and data types. Use `null` or empty lists (`[]`) where information is not found or not applicable.
5.  **Generate Single JSON:** Ensure the final output is one complete, valid JSON object.

**Desired JSON Schema:**

```json
{
  "job_title": "string", // The primary job title mentioned (e.g., "Software Engineer", "Data Analyst").
  "company_name": "string_or_null", // The name of the hiring company, or null if not specified.
  "location": "string_or_null", // The primary work location (e.g., "Remote", "New York, NY", "Hybrid - London"), or null if not specified.
  "required_skills": [
    // List of ALL skills (technical/hard skills) explicitly required.
    // Example: ["Python", "SQL", "AWS"]
    // Use keywords only. Empty list [] if none are explicitly required.
    "string"
   ],
  "preferred_skills": [
    // List of ALL skills (technical/hard skills) mentioned as preferred, a plus, or nice-to-have.
    // Example: ["Java",  "Kubernetes",  "GCP"]
    // Use keywords only. Empty list [] if none are mentioned.
    "string"
  ],
  "required_experience_years": "integer_or_null", // Minimum years of *relevant* experience explicitly required (e.g., if text says "5+ years" or "minimum 5 years", use 5). Use null if no specific minimum duration is mentioned.
  "key_responsibilities": [
    // List of main duties, tasks, and responsibilities described for the role.
    // Extract each distinct responsibility as a concise string.
    // Use keywords as well like Python,AWS etc.
    // Example: ["Develop backend services using Flask", "Collaborate with product managers", "Write unit tests"]
    // Empty list [] if none are listed.
    "string"
  ],
  "other_qualifications": [
     // List of other beneficial qualifications, experiences, or attributes not categorized as skills (required or preferred).
     // Includes: Certifications (e.g., "PMP Certification", "AWS Certified Developer"), specific industry experience (e.g., "FinTech experience", "Healthcare background"), educational degrees (if mentioned as beneficial beyond a baseline requirement), security clearance, etc.
     // Example: ["B.S. in Computer Science", "Experience in Agile environments", "Security Clearance Eligible"]
     // Empty list [] if none apply.
    "string"
  ]
}"""