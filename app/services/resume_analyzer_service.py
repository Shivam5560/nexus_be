import spacy
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class PracticalResumeAnalyzer:
    def __init__(self):
        # Load existing NLP model (spaCy)
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            # Fallback to smaller model if medium model not available
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("Warning: Using smaller language model. For better results, install en_core_web_md")
            except OSError:
                print("Please install spaCy models: python -m spacy download en_core_web_sm")
                raise

        # Action verbs dictionary - instead of a pre-trained model
        self.action_verbs = self._load_action_verbs()

        # Industry-specific skill lists
        self.industry_skills = self._load_industry_skills()

        # Initialize TF-IDF vectorizer for text analysis
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def _load_action_verbs(self):
        """Load action verbs dictionary with impact ratings"""
        # Strong action verbs get higher ratings
        high_impact = {
            "achieved": 1.0, "accelerated": 0.9, "spearheaded": 1.0, "pioneered": 0.9,
            "transformed": 1.0, "revolutionized": 0.9, "maximized": 0.9, "orchestrated": 0.9,
            "launched": 0.9, "created": 0.8, "developed": 0.8, "implemented": 0.8,
            "increased": 0.9, "decreased": 0.9, "reduced": 0.9, "improved": 0.8,
            "generated": 0.9, "delivered": 0.8, "produced": 0.8, "designed": 0.8,
            "led": 0.9, "managed": 0.8, "directed": 0.8, "supervised": 0.7,
            "coordinated": 0.7, "executed": 0.7, "established": 0.8, "formulated": 0.8
        }

        medium_impact = {
            "administered": 0.6, "analyzed": 0.7, "approved": 0.6, "arranged": 0.5,
            "collaborated": 0.7, "conducted": 0.6, "consolidated": 0.7, "controlled": 0.6,
            "delegated": 0.7, "evaluated": 0.7, "facilitated": 0.6, "identified": 0.7,
            "monitored": 0.6, "operated": 0.5, "organized": 0.6, "planned": 0.6,
            "prepared": 0.5, "processed": 0.5, "recommended": 0.7, "reviewed": 0.6,
            "scheduled": 0.5, "streamlined": 0.7, "updated": 0.5, "utilized": 0.5
        }

        low_impact = {
            "assisted": 0.4, "contributed": 0.4, "participated": 0.3, "helped": 0.3,
            "handled": 0.4, "maintained": 0.4, "provided": 0.4, "supported": 0.4,
            "used": 0.2, "worked": 0.2, "responsible": 0.2, "involved": 0.2,
            "included": 0.2, "performed": 0.4, "served": 0.3, "completed": 0.4
        }

        # Combine all verbs
        all_verbs = {}
        all_verbs.update(high_impact)
        all_verbs.update(medium_impact)
        all_verbs.update(low_impact)

        return all_verbs

    def _load_industry_skills(self):
        """Load industry-specific skills lists"""
        return {
            "tech": [
                "python", "java", "javascript", "react", "angular", "vue", "node.js", "typescript",
                "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "devops", "agile", "scrum",
                "machine learning", "ai", "data science", "big data", "sql", "nosql", "mongodb",
                "git", "rest api", "microservices", "cloud", "web development", "mobile development"
            ],
            "finance": [
                "financial analysis", "financial modeling", "accounting", "bookkeeping", "quickbooks",
                "sap", "excel", "financial reporting", "budgeting", "forecasting", "investment",
                "portfolio management", "risk assessment", "compliance", "regulatory", "banking",
                "insurance", "underwriting", "taxation", "audit", "cpa", "cfa", "fintech"
            ],
            "healthcare": [
                "patient care", "clinical", "medical", "healthcare", "nursing", "physician",
                "pharmacology", "ehr", "epic", "cerner", "hipaa", "medical coding", "medical billing",
                "patient records", "clinical trials", "research", "diagnostics", "treatment",
                "care coordination", "telehealth", "public health", "medical devices"
            ],
            "marketing": [
                "digital marketing", "social media", "seo", "sem", "content marketing", "copywriting",
                "brand management", "market research", "analytics", "campaign management", "crm",
                "hubspot", "salesforce", "mailchimp", "google analytics", "google ads", "facebook ads",
                "instagram", "tiktok", "public relations", "advertising", "graphic design"
            ]
        }

    def detect_passive_voice(self, sentences):
        """Uses spaCy to detect passive voice constructions"""
        passive_count = 0
        total_sentences = len(sentences)
        passive_sentences = []

        for sent in sentences:
            doc = self.nlp(sent)

            # Look for passive voice patterns using dependency parsing
            is_passive = False
            for token in doc:
                if "pass" in token.dep_:  # Look for nsubjpass, auxpass, etc.
                    is_passive = True
                    passive_sentences.append(sent)
                    break

            if is_passive:
                passive_count += 1

        passive_ratio = passive_count / total_sentences if total_sentences else 0

        if passive_ratio > 0.3:
            score = 0.6
            recommendation = "Too many passive voice constructions detected. Use active voice for stronger impact."
            if passive_sentences:
                recommendation += f" Example: \"{passive_sentences[0]}\""
        elif passive_ratio > 0.1:
            score = 0.8
            recommendation = "Some passive voice detected. Consider revising for more active tone."
        else:
            score = 1.0
            recommendation = None

        return score, recommendation

    def analyze_action_verbs(self, bullet_points):
        """Analyzes the quality and frequency of action verbs using our dictionary"""
        if not bullet_points:
            return 0.5, ["Work experience section missing or lacks proper bullet points."]

        verb_scores = []
        weak_bullets = []
        missing_verbs = []

        for bullet in bullet_points:
            # Skip empty bullets
            if not bullet.strip():
                continue

            doc = self.nlp(bullet.strip())

            # Check first word to see if it's an action verb
            if doc and len(doc) > 0:
                first_word = doc[0].lemma_.lower()

                if first_word in self.action_verbs:
                    verb_scores.append(self.action_verbs[first_word])

                    # Track weak verbs
                    if self.action_verbs[first_word] < 0.5:
                        weak_bullets.append((bullet, first_word))
                else:
                    # Not starting with an action verb
                    missing_verbs.append(bullet)

        # Calculate scores and generate recommendations
        recommendations = []

        # Check if bullets start with action verbs
        if missing_verbs:
            missing_ratio = len(missing_verbs) / len(bullet_points)
            if missing_ratio > 0.3:
                recommendations.append(
                    f"Many bullet points ({int(missing_ratio * 100)}%) don't start with action verbs. Start each bullet with a strong action verb.")
                if missing_verbs:
                    recommendations.append(f"Example bullet to improve: \"{missing_verbs[0]}\"")

        # Check strength of action verbs
        if verb_scores:
            avg_verb_score = sum(verb_scores) / len(verb_scores)

            if avg_verb_score < 0.5:
                recommendations.append("Using weak action verbs. Replace with stronger, more impactful verbs.")
                if weak_bullets:
                    bullet, verb = weak_bullets[0]
                    better_verbs = [v for v, s in self.action_verbs.items() if s > 0.8][:3]
                    recommendations.append(
                        f"Example: Replace \"{verb}\" in \"{bullet}\" with stronger alternatives like {', '.join(better_verbs)}.")
            elif avg_verb_score < 0.7:
                recommendations.append("Consider using more powerful action verbs for greater impact.")

            # Check verb diversity
            used_verbs = [doc[0].lemma_.lower() for bullet in bullet_points
                          if bullet.strip() and self.nlp(bullet.strip())[0].lemma_.lower() in self.action_verbs]
            unique_verbs = set(used_verbs)

            if len(used_verbs) > 5 and len(unique_verbs) / len(used_verbs) < 0.6:
                recommendations.append(
                    "Low diversity of action verbs. Vary your word choice to highlight different skills.")

            # Calculate final score
            if missing_ratio > 0.3:
                score = 0.6
            elif avg_verb_score < 0.5:
                score = 0.7
            elif avg_verb_score < 0.7:
                score = 0.8
            else:
                score = 1.0

        else:
            score = 0.5
            recommendations.append("No action verbs detected. Start each bullet point with a strong action verb.")

        return score, recommendations

    def detect_quantifiable_achievements(self, bullet_points):
        """Identifies and evaluates quantifiable achievements"""
        if not bullet_points:
            return 0.6, "Work experience section missing or lacks proper bullet points."

        # Patterns for numbers, currencies, percentages
        number_pattern = re.compile(r'\b\d+[%+]?\b|\b\d+\.\d+[%+]?\b|\b\d+\s*[kmbt]\b|\$\d+')
        percentage_pattern = re.compile(r'\b\d+(\.\d+)?%')
        currency_pattern = re.compile(r'\$\d+(\,\d+)*(\.\d+)?|\b\d+(\.\d+)?\s*(dollars|USD|EUR|GBP)\b')
        time_pattern = re.compile(r'\b\d+\s*(day|week|month|year|hour)s?\b')

        # Impact indicators
        impact_indicators = ["increased", "decreased", "reduced", "improved", "generated",
                             "saved", "exceeded", "grew", "led", "managed", "delivered"]

        quantifiable_count = 0
        impact_count = 0
        percentage_count = 0
        currency_count = 0
        time_count = 0

        non_quantified = []

        for bullet in bullet_points:
            bullet_lower = bullet.lower()

            # Check for numbers
            has_number = bool(number_pattern.search(bullet))
            has_percentage = bool(percentage_pattern.search(bullet))
            has_currency = bool(currency_pattern.search(bullet))
            has_time = bool(time_pattern.search(bullet))

            # Check for impact verbs
            has_impact = any(indicator in bullet_lower for indicator in impact_indicators)

            if has_number:
                quantifiable_count += 1

                if has_percentage:
                    percentage_count += 1
                if has_currency:
                    currency_count += 1
                if has_time:
                    time_count += 1

                if has_impact:
                    impact_count += 1
            else:
                non_quantified.append(bullet)

        # Calculate metrics
        total_bullets = len(bullet_points)
        quant_ratio = quantifiable_count / total_bullets if total_bullets else 0
        impact_ratio = impact_count / total_bullets if total_bullets else 0

        # Generate recommendations
        if quant_ratio < 0.2:
            score = 0.6
            recommendation = "Add more quantifiable achievements. Include numbers, percentages, or metrics to demonstrate impact."
            if non_quantified and len(non_quantified) > 0:
                recommendation += f" Example bullet to improve: \"{non_quantified[0]}\""
        elif quant_ratio < 0.4:
            score = 0.8
            recommendation = "Add a few more quantifiable achievements for maximum impact."
        else:
            if impact_ratio < 0.2:
                score = 0.9
                recommendation = "Good use of numbers, but link them more directly to your impact and contributions."
            else:
                score = 1.0
                recommendation = None

        # If using only one type of metric, suggest diversity
        if quantifiable_count > 3:
            metric_types = sum(x > 0 for x in [percentage_count, currency_count, time_count])
            if metric_types == 1:
                if recommendation:
                    recommendation += " Try using a diverse range of metrics (percentages, currency, time)."
                else:
                    score = 0.9
                    recommendation = "Consider using a more diverse range of metrics (percentages, currency, time)."

        return score, recommendation

    def analyze_industry_fit(self, resume_text, industry="default"):
        """Analyzes how well the resume matches the target industry"""
        if industry == "default" or industry not in self.industry_skills:
            return 1.0, None

        # Create a set of skills for the given industry
        industry_skills_set = set(skill.lower() for skill in self.industry_skills[industry])

        # Find matches in the resume text
        resume_lower = resume_text.lower()
        matched_skills = [skill for skill in industry_skills_set if skill in resume_lower]

        # Calculate match ratio
        match_ratio = len(matched_skills) / len(industry_skills_set)

        # Generate recommendation based on match
        if match_ratio < 0.1:
            score = 0.5
            recommendation = f"Your resume shows very few {industry} industry-specific skills. Consider adding relevant keywords."
        elif match_ratio < 0.2:
            score = 0.7
            recommendation = f"Add more {industry}-specific skills and keywords to better target this industry."
        elif match_ratio < 0.3:
            score = 0.8
            recommendation = f"Consider highlighting a few more {industry}-specific skills."
        else:
            score = 1.0
            recommendation = None

        return score, recommendation

    def analyze_resume(self, resume_text, resume_dict, industry="default"):
        """Main analysis function that uses NLP and pattern recognition"""
        # Parse resume with spaCy
        doc = self.nlp(resume_text)
        sentences = [sent.text for sent in doc.sents]

        # Extract work experience bullet points
        work_experience = resume_dict.get('work_experience', [])
        all_bullets = []
        for exp in work_experience:
            all_bullets.extend(exp.get('responsibilities', []))

        # Run individual analyses
        section_scores = {}
        all_recommendations = []

        # 1. Length assessment using semantic density
        # print("1")
        length_score, length_rec = self._analyze_length(doc)
        section_scores["length"] = round(length_score * 100, 2)
        if length_rec:
            all_recommendations.append(length_rec)

        # 2. Action verb analysis
        # print("2")

        verb_score, verb_recs = self.analyze_action_verbs(all_bullets)
        section_scores["action_verbs"] = round(verb_score * 100, 2)
        all_recommendations.extend(verb_recs)

        # 3. Bullet point quality and format
        # print("3")

        bullet_score, bullet_rec = self._analyze_bullet_quality(all_bullets, work_experience)
        section_scores["bullet_points"] = round(bullet_score * 100, 2)
        if bullet_rec:
            all_recommendations.append(bullet_rec)

        # 4. Quantifiable achievements
        # print("4")
        quant_score, quant_rec = self.detect_quantifiable_achievements(all_bullets)
        section_scores["quantifiable"] = round(quant_score * 100, 2)
        if quant_rec:
            all_recommendations.append(quant_rec)

        # 5. Sentence structure analysis
        # print("5")
        sent_score, sent_rec = self._analyze_sentence_structure(sentences)
        section_scores["sentence_structure"] = round(sent_score * 100, 2)
        if sent_rec:
            all_recommendations.append(sent_rec)

        # 6. Passive voice detection
        # print("6")
        passive_score, passive_rec = self.detect_passive_voice(sentences)
        section_scores["active_voice"] = round(passive_score * 100, 2)
        if passive_rec:
            all_recommendations.append(passive_rec)

        # 7. Section completeness
        # print("7")
        completeness_score, completeness_rec = self._analyze_section_completeness(resume_dict)
        section_scores["completeness"] = round(completeness_score * 100, 2)
        if completeness_rec:
            all_recommendations.append(completeness_rec)

        # 8. Skills analysis
        # print("8")
        skills_score, skills_rec = self._analyze_skills(resume_dict.get('skills', []))
        section_scores["skills_format"] = round(skills_score * 100, 2)
        if skills_rec:
            all_recommendations.append(skills_rec)

        # # 9. Contact info analysis
        # print("9")
        # contact_score, contact_rec = self._analyze_contact_info(resume_dict.get('personal_info', {}))
        # section_scores["contact_info"] = round(contact_score * 100, 2)
        # if contact_rec:
        #     all_recommendations.append(contact_rec)

        # 10. Industry fit analysis
        # print("10")
        industry_score, industry_rec = self.analyze_industry_fit(resume_text, industry)
        section_scores["industry_fit"] = round(industry_score * 100, 2)
        if industry_rec:
            all_recommendations.append(industry_rec)

        # Calculate overall score
        weights = {
            "length": 0.05,
            "action_verbs": 0.15,
            "bullet_points": 0.1,
            "quantifiable": 0.15,
            "sentence_structure": 0.10,
            "active_voice": 0.1,
            "completeness": 0.1,
            "skills_format": 0.1,
            "industry_fit": 0.15
        }

        weighted_score = sum(section_scores[key] * weights[key] for key in weights)

        return round(weighted_score, 2), all_recommendations, section_scores

    def _analyze_length(self, doc):
        """Analyzes resume length based on meaningful content"""
        # Count tokens excluding stopwords and punctuation
        meaningful_tokens = [token for token in doc if not token.is_stop and not token.is_punct]
        meaningful_count = len(meaningful_tokens)

        if meaningful_count < 200:
            return 0.6, "Resume lacks sufficient detail. Aim for 400-800 words to adequately showcase your experience."
        elif meaningful_count > 800:
            return 0.8, "Resume is too dense. Focus on most relevant information for better readability."
        else:
            return 1.0, None

    def _analyze_bullet_quality(self, all_bullets, work_experience):
        """Analyzes bullet point quality and distribution"""
        if not work_experience or not all_bullets:
            return 0.5, "Work experience section missing or lacks proper bullet points."

        # Check bullet point distribution across jobs
        jobs_with_bullets = sum(1 for exp in work_experience if exp.get('responsibilities', []))
        total_jobs = len(work_experience)

        if jobs_with_bullets < total_jobs:
            return 0.7, "Some positions lack bullet points. Add achievements for all roles."

        # Calculate bullet points per job
        bullets_per_job = [len(exp.get('responsibilities', [])) for exp in work_experience]
        if not bullets_per_job:
            return 0.5, "Work experience section lacks proper formatting with bullet points."

        avg_bullets = sum(bullets_per_job) / len(bullets_per_job)

        # Check bullet point length (by words)
        bullet_lengths = [len(bullet.split()) for bullet in all_bullets]
        avg_length = sum(bullet_lengths) / len(bullet_lengths) if bullet_lengths else 0

        if avg_bullets < 2:
            return 0.5, "Too few bullet points per role. Aim for 3-5 achievements per position."
        elif avg_bullets > 8:
            return 0.7, "Too many bullet points per role. Focus on 3-5 key achievements."
        elif avg_length < 6:
            return 0.8, "Bullet points are too brief. Provide more context about your achievements."
        elif avg_length > 25:
            return 0.8, "Some bullet points are too long. Keep them concise (1-2 lines each)."
        else:
            return 1.0, None

    def _analyze_sentence_structure(self, sentences):
        """Analyzes sentence structure and complexity"""
        if not sentences:
            return 0.7, "Resume format issue. Ensure proper sentence structure."

        # Calculate sentence lengths
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

        # Calculate complexity through average token length
        complexity = 0
        total_words = 0
        for sent in sentences:
            doc = self.nlp(sent)
            words = [token.text for token in doc if not token.is_punct]
            total_words += len(words)
            complexity += sum(len(word) for word in words)

        avg_word_length = complexity / total_words if total_words > 0 else 0

        if avg_length > 25:
            return 0.6, "Sentences are too long. Aim for 15-20 words per sentence for better readability."
        elif avg_length < 5 and len(sentences) > 5:
            return 0.7, "Some sentences are too short. Combine brief phrases into more substantial points."
        elif avg_word_length > 7:
            return 0.8, "Your vocabulary may be too complex. Use more straightforward language."
        else:
            return 1.0, None

    def _analyze_section_completeness(self, resume_dict):
        """Analyzes resume section completeness"""
        essential_sections = ['personal_info', 'education', 'work_experience', 'skills']
        recommended_sections = ['certifications', 'projects', 'summary', 'achievements']

        # Check which sections are present
        present_essential = [section for section in essential_sections if resume_dict.get(section)]
        present_recommended = [section for section in recommended_sections if resume_dict.get(section)]

        # Calculate completeness
        essential_ratio = len(present_essential) / len(essential_sections)
        recommended_ratio = len(present_recommended) / len(recommended_sections) if recommended_sections else 1.0

        # Weight essential sections more heavily
        completeness_score = essential_ratio * 0.8 + recommended_ratio * 0.2

        if essential_ratio < 1.0:
            missing = [s for s in essential_sections if s not in present_essential]
            return 0.7, f"Missing essential sections: {', '.join(missing)}. Include all key resume components."
        elif recommended_ratio < 0.25:
            return 0.9, "Consider adding additional sections like summary, certifications, or projects."
        else:
            return 1.0, None

    def _analyze_skills(self, skills):
        """Analyzes skills section format and content"""
        if not skills:
            return 0.5, "Skills section missing or empty. Include a dedicated skills section."

        # Check skills count
        if len(skills) < 5:
            return 0.6, "Too few skills listed. Include 8-12 relevant skills."
        elif len(skills) > 20:
            return 0.8, "Too many skills listed. Focus on 8-15 most relevant skills."

        # Check skill phrase length (avoid long skill descriptions)
        long_skills = [skill for skill in skills if len(skill.split()) > 3]
        if long_skills and len(long_skills) / len(skills) > 0.3:
            return 0.8, "Some skills are too verbose. Keep skills concise (1-3 words each)."

        return 1.0, None

    def _analyze_contact_info(self, personal_info):
        """Analyzes contact information completeness"""
        if not personal_info:
            return 0.5, "Missing personal information section with contact details."

        required_fields = ['name', 'email', 'phone']
        present_fields = [field for field in required_fields if personal_info.get(field)]

        if len(present_fields) < len(required_fields):
            missing = [f for f in required_fields if f not in present_fields]
            return 0.7, f"Complete your contact information by adding: {', '.join(missing)}"

        # Check for LinkedIn profile
        if not personal_info.get('linkedin'):
            return 0.9, "Consider adding your LinkedIn profile URL to your contact information."

        return 1.0, None

    def generate_summary_report(self, score, recommendations, section_scores):
        """Generates a readable summary report of the analysis"""
        report = []

        # Overall score
        report.append(f"Overall Resume Score: {score}/100")

        if score >= 90:
            report.append("\nYour resume is excellent! Here are a few final touches to perfect it:")
        elif score >= 80:
            report.append("\nYour resume is strong, but could benefit from these improvements:")
        elif score >= 70:
            report.append("\nYour resume needs improvement in several areas:")
        else:
            report.append("\nYour resume requires significant improvement. Focus on these areas:")

        # Section scores from highest to lowest
        report.append("\nSection Scores:")
        sorted_scores = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
        for section, score in sorted_scores:
            section_name = section.replace("_", " ").title()
            report.append(f"- {section_name}: {score}/100")

        # Key recommendations
        report.append("\nKey Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            report.append(f"{i}. {rec}")

        # Additional recommendations count
        if len(recommendations) > 5:
            report.append(f"\nPlus {len(recommendations) - 5} more recommendations.")

        return "\n".join(report)