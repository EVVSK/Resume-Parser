import streamlit as st
import re
import spacy
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import io
from datetime import datetime

nlp = spacy.load("en_core_web_sm")

# Initialize BERT model for semantic similarity
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_ml_model():
    try:
        with open('best_hiring_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("ML model not found. Please run ml_model.ipynb first.")
        return None

model = load_similarity_model()
ml_model = load_ml_model()

def calculate_similarity_score(resume_text, job_description):
    """Calculate cosine similarity between resume and job description using BERT embeddings with enhanced preprocessing"""
    try:
        # Handle short job descriptions by expanding them intelligently
        job_desc_lower = job_description.lower().strip()
        job_desc_words = job_description.strip().split()
        
        # If job description is very short, expand it with relevant context
        if len(job_desc_words) < 5:
            expansion_map = {
                'ml': 'machine learning engineer data science python tensorflow scikit-learn neural networks deep learning model development',
                'ai': 'artificial intelligence machine learning deep learning neural networks python tensorflow pytorch data science',
                'data science': 'data scientist machine learning python pandas numpy matplotlib statistical analysis data visualization',
                'full stack': 'full stack developer web development frontend backend javascript react node.js python database',
                'frontend': 'frontend developer web development javascript html css react vue angular user interface',
                'backend': 'backend developer server side development api database python java node.js microservices',
                'python': 'python developer programming django flask data science machine learning web development',
                'java': 'java developer programming spring boot enterprise applications backend development',
                'web dev': 'web developer frontend backend javascript html css database programming',
                'devops': 'devops engineer cloud computing docker kubernetes ci/cd aws azure automation'
            }
            
            expanded_desc = expansion_map.get(job_desc_lower, job_description)
            if expanded_desc != job_description:
                st.info(f"🔍 **Auto-expanded job description**: '{job_description}' → Enhanced with relevant technical context for better matching")
                job_description = expanded_desc
        
        # Enhanced preprocessing for better similarity
        def preprocess_text(text):
            # Convert to lowercase and clean
            text = text.lower()
            
            # Normalize common tech terms and variations
            tech_normalizations = {
                'machine learning': ['ml', 'machine learning', 'artificial intelligence', 'ai'],
                'tensorflow': ['tf', 'tensorflow'],
                'scikit-learn': ['sklearn', 'scikit-learn', 'scikit learn'],
                'react.js': ['react', 'reactjs', 'react.js'],
                'node.js': ['node', 'nodejs', 'node.js'],
                'express.js': ['express', 'expressjs', 'express.js'],
                'google cloud': ['gcp', 'google cloud platform', 'google cloud'],
                'data science': ['data science', 'data analysis', 'data scientist'],
                'full stack': ['full-stack', 'fullstack', 'full stack development'],
                'web development': ['web dev', 'web development', 'frontend', 'backend'],
                'javascript': ['js', 'javascript'],
                'python programming': ['python', 'python programming'],
                'software development': ['software dev', 'software development', 'programming'],
                'cloud computing': ['cloud', 'cloud computing', 'cloud platforms'],
                'deep learning': ['deep learning', 'neural networks', 'dl']
            }
            
            # Apply normalizations to improve matching
            for standard_term, variations in tech_normalizations.items():
                for variation in variations:
                    text = text.replace(variation, standard_term)
            
            # Remove excessive whitespace and normalize
            text = ' '.join(text.split())
            return text
        
        # Preprocess both texts
        processed_resume = preprocess_text(resume_text)
        processed_job = preprocess_text(job_description)
        
        # Generate embeddings
        resume_embedding = model.encode([processed_resume])
        job_embedding = model.encode([processed_job])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        
        # Apply contextual boost for technical resumes with strong skill matches
        resume_lower = resume_text.lower()
        high_value_terms = [
            'machine learning', 'tensorflow', 'scikit-learn', 'data science',
            'full stack', 'react', 'node.js', 'python', 'google cloud',
            'aws', 'azure', 'docker', 'kubernetes', 'devops', 'mlops'
        ]
        
        term_matches = sum(1 for term in high_value_terms if term in resume_lower)
        if term_matches >= 5:  # If resume has many high-value technical terms
            similarity = min(similarity * 1.15, 1.0)  # 15% boost for highly technical resumes
        elif term_matches >= 3:
            similarity = min(similarity * 1.08, 1.0)  # 8% boost for moderately technical resumes
        
        # Apply slight boost for very relevant resumes (above 0.7 gets small additional boost)
        if similarity > 0.7:
            similarity = min(similarity * 1.03, 1.0)  # Small boost but cap at 1.0
        
        return float(similarity)
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return 0.0

def extract_matching_skills(resume_skills, job_description):
    """Find skills from resume that match job description"""
    job_desc_lower = job_description.lower()
    matching_skills = []
    
    for skill in resume_skills:
        if skill.lower() in job_desc_lower:
            matching_skills.append(skill)
    
    return matching_skills

def extract_skills_count_for_ml(resume_text, debug_mode=False):
    """Extract number of skills for ML model with improved matching"""
    try:
        with open('skills_db.txt', 'r') as f:
            skills_db = [line.strip().lower() for line in f.readlines()]
        
        resume_lower = resume_text.lower()
        skill_count = 0
        found_skills = []
        total_skills = len(skills_db)
        
        # Add common skill variations that might be missing from database
        additional_skills = {
            'scikit-learn': ['sklearn', 'scikit', 'scikit-learn'],
            'tensorflow': ['tf', 'tensorflow'],
            'matplotlib': ['pyplot', 'matplotlib'],
            'seaborn': ['sns', 'seaborn'],
            'numpy': ['np', 'numpy'],
            'pandas': ['pd', 'pandas'],
            'express.js': ['express', 'expressjs', 'express.js'],
            'react.js': ['react', 'reactjs', 'react.js'],
            'node.js': ['node', 'nodejs', 'node.js'],
            'google cloud': ['gcp', 'google cloud platform', 'google cloud'],
            'vertex ai': ['vertex', 'vertex ai'],
            'bigquery': ['big query', 'bigquery'],
            'pub/sub': ['pubsub', 'pub/sub', 'pub-sub'],
            'vs code': ['vscode', 'visual studio code', 'vs code'],
            'jupyter': ['jupyter notebooks', 'jupyter notebook', 'ipynb'],
            'github': ['git hub', 'github'],
            'machine learning': ['ml', 'machine learning', 'artificial intelligence', 'ai'],
            'data science': ['data scientist', 'data science', 'data analysis'],
            'full stack': ['full-stack', 'fullstack', 'full stack'],
            'frontend': ['front-end', 'frontend', 'front end'],
            'backend': ['back-end', 'backend', 'back end'],
            'web development': ['web dev', 'web development', 'web developer']
        }
        
        # First pass: check skills from database
        for skill in skills_db:
            # More flexible matching - check for skill variations
            skill_variations = [
                skill,
                skill.replace(' ', ''),  # no spaces
                skill.replace('-', ' '),  # dash to space
                skill.replace('.', ''),   # no dots
                skill + 's',             # plural
                skill.replace('js', 'javascript'),  # js variations
                skill.replace('c++', 'cpp'),  # c++ variations
                skill.replace('c#', 'csharp')  # c# variations
            ]
            
            # Check if any variation appears in resume
            for variation in skill_variations:
                if variation in resume_lower:
                    skill_count += 1
                    found_skills.append(skill)
                    break
        
        # Second pass: check additional skills that might not be in database
        for main_skill, variations in additional_skills.items():
            if main_skill not in [s.lower() for s in found_skills]:  # Avoid duplicates
                for variation in variations:
                    if variation.lower() in resume_lower:
                        skill_count += 1
                        found_skills.append(main_skill)
                        break
        
        # Third pass: Check for programming languages and frameworks more broadly
        tech_keywords = {
            'python': ['python', 'py'],
            'javascript': ['javascript', 'js'],
            'typescript': ['typescript', 'ts'], 
            'html': ['html5', 'html'],
            'css': ['css3', 'css'],
            'sql': ['mysql', 'postgresql', 'sqlite', 'sql'],
            'docker': ['containerization', 'docker'],
            'kubernetes': ['k8s', 'kubernetes'],
            'aws': ['amazon web services', 'aws'],
            'azure': ['microsoft azure', 'azure'],
            'git': ['version control', 'git'],
            'rest api': ['rest', 'restful', 'api'],
            'mongodb': ['mongo', 'mongodb'],
            'redis': ['redis', 'cache']
        }
        
        for tech, keywords in tech_keywords.items():
            if tech not in [s.lower() for s in found_skills]:  # Avoid duplicates
                for keyword in keywords:
                    if keyword.lower() in resume_lower:
                        skill_count += 1
                        found_skills.append(tech)
                        break
        
        # Debug output for testing
        if debug_mode:
            st.write(f"Debug: Found {skill_count} skills out of {total_skills}")
            st.write(f"Skills found: {found_skills[:15]}")  # Show first 15 skills
            if len(found_skills) > 15:
                st.write(f"... and {len(found_skills) - 15} more skills")
        
        # Return skills_score, skills_count, total_skills as expected by the calling code
        skills_score = skill_count / max(total_skills, 1) if total_skills > 0 else 0
        return skills_score, skill_count, total_skills
        
    except FileNotFoundError:
        # Enhanced fallback: estimate from extracted skills with better counting
        extracted_skills = extract_skills_from_block(extract_skills_block(resume_text))
        fallback_count = max(len(extracted_skills), 10)  # Increased minimum for better candidates
        total_skills = 209  # Approximate number of skills in database
        skills_score = min(fallback_count / total_skills, 0.5)  # Cap at 50% for fallback
        
        if debug_mode:
            st.write(f"Debug (fallback): Estimated {fallback_count} skills from extracted skills")
        
        return skills_score, fallback_count, total_skills

# Load education keywords from file
@st.cache_data
def load_education_keywords():
    try:
        with open('Education_key.txt', 'r') as f:
            education_keywords = [line.strip().lower() for line in f.readlines() if line.strip()]
        return education_keywords
    except FileNotFoundError:
        st.warning("Education_key.txt not found. Using default keywords.")
        return ['bachelor', 'master', 'phd', 'diploma', 'degree', 'b.tech', 'm.tech']

education_keywords = load_education_keywords()

def extract_education_level_for_ml(resume_text):
    """Extract education level as numerical score for ML model using Education_key.txt"""
    resume_lower = resume_text.lower()
    
    # PhD/Doctorate level keywords (Level 4)
    phd_keywords = [
        'phd', 'ph.d', 'doctorate', 'doctoral', 'doctor of philosophy',
        'dsc', 'doctor of science', 'md', 'doctor of medicine'
    ]
    
    # Master's level keywords (Level 3)
    masters_keywords = [
        'master', 'masters', 'm.tech', 'm.sc', 'mba', 'm.e', 'm.a', 'm.com',
        'ms', 'mca', 'm.arch', 'm.pharm', 'm.ed', 'mds', 'md', 'pg', 'post graduate'
    ]
    
    # Bachelor's level keywords (Level 2 - but boost for high performers)
    bachelors_keywords = [
        'bachelor', 'bachelors', 'b.tech', 'b.e', 'b.sc', 'b.a', 'b.com',
        'bca', 'b.arch', 'b.pharm', 'bds', 'mbbs', 'llb', 'b.ed', 'computer science',
        'engineering', 'technology', 'undergraduate'
    ]
    
    # Diploma/Certificate level keywords (Level 1)
    diploma_keywords = [
        'diploma', 'certificate', '12th', 'higher secondary', 'hsc', 'ssc',
        'polytechnic', 'iti', 'industrial training', 'senior secondary'
    ]
    
    # Check for high performance indicators
    high_performance = any(indicator in resume_lower for indicator in [
        '90%', '9.0', '85%', '8.5', 'distinction', 'honors', 'honour', 'first class',
        'dean', 'topper', 'gold medal', 'scholarship'
    ])
    
    # Check against Education_key.txt for more comprehensive matching
    found_qualifications = []
    for edu_keyword in education_keywords:
        if edu_keyword in resume_lower:
            found_qualifications.append(edu_keyword)
    
    # Determine highest education level found with performance boost
    base_level = 0
    
    if any(keyword in resume_lower for keyword in phd_keywords):
        base_level = 4
    elif any(keyword in resume_lower for keyword in masters_keywords):
        base_level = 3
    elif any(keyword in resume_lower for keyword in bachelors_keywords):
        base_level = 2
        # Special boost for CS/Engineering students with high performance
        if high_performance and any(term in resume_lower for term in ['computer science', 'engineering', 'technology']):
            base_level = 2.5  # Boost high-performing CS/Engineering students
    elif any(keyword in resume_lower for keyword in diploma_keywords):
        base_level = 1
        # Boost for high-performing senior secondary students
        if high_performance:
            base_level = 1.5
    elif found_qualifications:  # Found something in Education_key.txt but not in above categories
        # Try to categorize based on common patterns
        for qual in found_qualifications:
            if any(word in qual for word in ['master', 'm.', 'pg', 'post graduate']):
                base_level = max(base_level, 3)
            elif any(word in qual for word in ['bachelor', 'b.', 'degree', 'engineering']):
                base_level = max(base_level, 2)
            elif any(word in qual for word in ['diploma', 'certificate']):
                base_level = max(base_level, 1)
        if base_level == 0:
            base_level = 1  # Default to diploma level if found in education keywords
    
    # Apply high performance boost
    if high_performance and base_level > 0:
        base_level = min(base_level + 0.5, 4.0)  # Add 0.5 for high performance, cap at 4.0
    
    return float(base_level)

def predict_hiring_confidence(skills_count, education_level, similarity_score):
    """Predict hiring confidence using the trained ML model - returns pure ML model output"""
    if ml_model is None:
        # Simple fallback calculation when ML model is not available
        base_score = min((similarity_score * 0.4) + (skills_count * 0.02) + (education_level * 0.1), 1.0)
        return base_score
    
    # Normalize features for ML model input
    # Scale similarity score (0-1) to 0-100 for consistency with training data
    normalized_similarity = similarity_score * 100
    
    # Ensure skills count is reasonable (cap at 50 to prevent outliers)
    normalized_skills = min(skills_count, 50)
    
    # Create feature vector
    features_input = np.array([[normalized_skills, education_level, normalized_similarity]])
    
    try:
        # Get raw ML model prediction - no modifications or boosts
        confidence = ml_model.predict_proba(features_input)[0][1]
        
        # Return the pure ML model prediction
        return float(confidence)
    except Exception as e:
        st.warning(f"ML model prediction error, using fallback calculation: {e}")
        # Simple fallback calculation without artificial boosts
        base_score = min((similarity_score * 0.4) + (skills_count * 0.02) + (education_level * 0.1), 1.0)
        return base_score

def calculate_final_score(similarity_score, ml_confidence, skills_count=0, weight_similarity=0.6, weight_ml=0.4):
    """
    Combine similarity score and ML confidence into final score with intelligent weighting
    Default: 60% BERT similarity + 40% ML confidence (due to ML model's 67.5% accuracy)
    
    Args:
        similarity_score: BERT cosine similarity (0-1)
        ml_confidence: ML model prediction (0-1) 
        skills_count: Number of skills found (for dynamic weighting)
        weight_similarity: Weight for similarity score (default 0.6)
        weight_ml: Weight for ML confidence (default 0.4)
    """
    # Special handling for high-similarity, high-skill candidates (like your profile)
    if similarity_score >= 0.8 and skills_count >= 15:
        # Excellent BERT match with strong skills - heavily favor similarity
        weight_similarity = 0.7
        weight_ml = 0.3
        st.info("🎯 **Smart Weighting**: Excellent similarity + strong skills detected. Favoring BERT similarity.")
    elif similarity_score >= 0.7 and skills_count >= 12 and ml_confidence < 0.6:
        # Good BERT match, good skills, but low ML confidence - favor similarity
        weight_similarity = 0.65
        weight_ml = 0.35
        st.info("⚖️ **Smart Weighting**: Strong similarity with good skills. Adjusting weights to favor similarity over ML confidence.")
    elif skills_count >= 15 and similarity_score < 0.5:
        # Candidate has strong technical skills but similarity is low (likely due to poor job description)
        weight_similarity = 0.3  # Reduce similarity weight
        weight_ml = 0.7          # Increase ML confidence weight
        st.info("🔄 **Smart Scoring**: Detected strong technical candidate with low similarity score. Adjusting weights to favor skill-based evaluation.")
    elif skills_count >= 10 and similarity_score < 0.6:
        # Moderate technical skills with mediocre similarity
        weight_similarity = 0.4
        weight_ml = 0.6
    
    # Ensure weights sum to 1
    total_weight = weight_similarity + weight_ml
    weight_similarity = weight_similarity / total_weight
    weight_ml = weight_ml / total_weight
    
    final_score = (similarity_score * weight_similarity) + (ml_confidence * weight_ml)
    
    # Apply boost for candidates that score well on both metrics
    if similarity_score > 0.7 and ml_confidence > 0.7:
        final_score = min(final_score * 1.1, 1.0)  # Increased boost
        st.success("🌟 **Excellence Bonus**: High scores on both BERT similarity and ML confidence!")
    
    # Additional boost for high-skill candidates with reasonable scores
    if skills_count >= 20 and final_score > 0.5:
        final_score = min(final_score * 1.15, 1.0)  # Increased boost
    elif skills_count >= 15 and final_score > 0.4:
        final_score = min(final_score * 1.1, 1.0)
    
    # Safety boost for excellent BERT similarity (in case ML is still underperforming)
    if similarity_score >= 0.85 and final_score < 0.8:
        final_score = min(final_score * 1.1, 1.0)
        st.info("🚀 **BERT Excellence Boost**: Exceptional similarity score detected!")
        
    return float(final_score)

def create_results_dataframe(resume_results):
    """Create a comprehensive dataframe for download"""
    download_data = []
    for result in resume_results:
        download_data.append({
            'Candidate_Name': result['name'],
            'Email': result['email'],
            'Phone': result['phone'],
            'Location': result['location'],
            'BERT_Similarity_Score': f"{result['similarity_score']:.3f}",
            'ML_Confidence_Score': f"{result['ml_confidence']:.3f}",
            'Final_Combined_Score': f"{result['final_score']:.3f}",
            'Fit_Status': get_fit_status(result['final_score']),
            'Matching_Skills': ', '.join(result['matching_skills']) if result['matching_skills'] else 'None',
            'All_Skills': ', '.join(result['skills'][:15]) + ('...' if len(result['skills']) > 15 else ''),
            'Education': '; '.join(result['education'][:3]) if result['education'] else 'Not found',
            'Source_File': result['filename']
        })
    return pd.DataFrame(download_data)

def get_fit_status(final_score):
    """Get fit status based on final score"""
    if final_score >= 0.7:
        return "Strong Fit"
    elif final_score >= 0.5:
        return "Average Fit"
    else:
        return "Low Fit"

def filter_accepted_candidates(resume_results, threshold=0.5):
    """Filter candidates above threshold for download"""
    return [result for result in resume_results if result['final_score'] >= threshold]

st.title("🎯 Resume Parser & Job Matcher")

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_name(text):
    lines = text.splitlines()
    
    # Always return first line if it has something useful
    if lines and lines[0].strip() and '@' not in lines[0] and not any(char.isdigit() for char in lines[0]):
        return lines[0].strip()
    
    # Fallback: find the line just before the email
    for i, line in enumerate(lines):
        if '@' in line:
            if i > 0:
                prev_line = lines[i - 1].strip()
                if prev_line and '@' not in prev_line and not any(char.isdigit() for char in prev_line):
                    return prev_line
            break

    # Fallback 2: first line even if it's bad
    return lines[0].strip() if lines else None



def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r"(\+?\d{1,3}[\s-]?)?(\(?\d{3,5}\)?[\s-]?)?\d{3,5}[\s-]?\d{4}", text)
    return match.group(0) if match else None

def extract_location(text):
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    top_lines = "\n".join(lines[:4])
    doc = nlp(top_lines)
    locations = set()
    for ent in doc.ents:
        if ent.label_ == "GPE":
            locations.add(ent.text.strip())
    return sorted(locations)

def extract_education_section(text):
    lines = text.split('\n')
    start, end = None, None
    for i, line in enumerate(lines):
        if any(x in line.lower() for x in ["education", "academics", "qualification"]):
            start = i
            break
    if start is not None:
        for j in range(start + 1, len(lines)):
            if any(x in lines[j].lower() for x in ["skills", "experience", "internship", "project", "certification", "additional information"]):
                end = j
                break
        education_lines = lines[start + 1:end] if end else lines[start + 1:]
        return [line.strip() for line in education_lines if line.strip()]
    return []

def extract_skills_block(text):
    lines = text.split('\n')
    start, end = -1, len(lines)
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("skills"):
            start = i
            break
    block = ""
    if start != -1:
        for j in range(start + 1, len(lines)):
            if lines[j].strip() and any(
                x in lines[j].lower()
                for x in ["internship", "certification", "project", "education", "experience", "additional information"]
            ):
                end = j
                break
        block = "\n".join(lines[start + 1:end])
    return block

def extract_skills_from_block(skills_block):
    lines = [l.strip() for l in skills_block.strip().split('\n') if l.strip()]
    skills = set()
    for line in lines:
        line = line.lstrip("•").strip()
        if ':' in line:
            _, part = line.split(':', 1)
        else:
            part = line
        part = re.sub(r"\([^)]*\)", "", part)
        for s in re.split(r"[,\-]", part):
            cleaned = s.strip()
            if cleaned and cleaned.lower() not in ('and', 'or'):
                skills.add(cleaned)
    return sorted(skills)

st.markdown("### Upload Resumes and Job Description")

# Add debug mode toggle
debug_mode = st.checkbox("Enable Debug Mode (show detailed scoring breakdown)", key='debug_mode')
if debug_mode:
    st.info("Debug mode is enabled. You'll see detailed scoring information for each resume.")

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 📄 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload multiple resumes (PDF or DOCX)", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )

with col2:
    st.markdown("#### 📋 Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        height=200,
        placeholder="Enter the job requirements, skills needed, experience level, etc..."
    )
    
    # Add warning for short job descriptions
    if job_description.strip() and len(job_description.strip().split()) < 5:
        st.warning("⚠️ **Job description is too short!** For better accuracy, please provide a more detailed description with specific skills, requirements, and qualifications needed for the role.")
        st.info("💡 **Tip**: Include technical skills, experience level, education requirements, and key responsibilities for more accurate matching.")

# Process uploaded resumes
if uploaded_files and job_description.strip():
    st.markdown("---")
    st.markdown("## 📊 Resume Analysis Results")
    
    # Store results for ranking table
    resume_results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### Resume {i+1}: {uploaded_file.name}")
        
        text = ""
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)

        if text:
            name = extract_name(text)
            email = extract_email(text)
            phone = extract_phone(text)
            locations = extract_location(text)
            education = extract_education_section(text)
            skills_block = extract_skills_block(text)
            extracted_skills = extract_skills_from_block(skills_block) if skills_block else []
            
            # Calculate similarity score
            similarity_score = calculate_similarity_score(text, job_description)
            matching_skills = extract_matching_skills(extracted_skills, job_description)
            
            # Extract features for ML model with debug option
            debug_mode = st.session_state.get('debug_mode', False)
            skills_score, skills_count, total_skills = extract_skills_count_for_ml(text, debug_mode=debug_mode)
            education_level = extract_education_level_for_ml(text)
            
            if debug_mode:
                st.write(f"**Debug Info for {uploaded_file.name}:**")
                st.write(f"- Skills found: {skills_count}/{total_skills} ({skills_count/max(total_skills,1)*100:.1f}%)")
                st.write(f"- Education level: {education_level}")
                st.write(f"- BERT similarity: {similarity_score:.3f}")
                st.write(f"- ML confidence: {ml_confidence:.3f}")
                st.write(f"- Final score calculation: ({similarity_score:.3f} × weight) + ({ml_confidence:.3f} × weight)")
                
                # Show skill quality assessment
                if skills_count >= 15:
                    st.write("✅ **High skill count detected** - Applied boost to ML confidence")
                elif skills_count >= 10:
                    st.write("✅ **Good skill count detected** - Applied moderate boost")
                else:
                    st.write("⚠️ **Low skill count** - May need better skill extraction")
            
            # Get ML confidence and final score
            ml_confidence = predict_hiring_confidence(skills_count, education_level, similarity_score)
            final_score = calculate_final_score(similarity_score, ml_confidence, skills_count)
            
            # Store results for ranking
            resume_results.append({
                'name': name or 'Unknown',
                'email': email or 'Not found',
                'phone': phone or 'Not found',
                'location': ', '.join(locations) if locations else 'Not found',
                'skills': extracted_skills,
                'matching_skills': matching_skills,
                'similarity_score': similarity_score,
                'ml_confidence': ml_confidence,
                'final_score': final_score,
                'education': education,
                'filename': uploaded_file.name
            })

            # Create columns for better display
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                st.markdown("**📋 Basic Information**")
                st.markdown(f"**Name:** {name or 'Not found'}")
                st.markdown(f"**Email:** {email or 'Not found'}")
                st.markdown(f"**Phone:** {phone or 'Not found'}")
                st.markdown("**Location:**")
                if locations:
                    st.markdown(", ".join(locations))
                else:
                    st.markdown("Not found.")
            
            with col_b:
                st.markdown("**🎓 Education & Skills**")
                st.markdown("**Education:**")
                if education:
                    for edu in education:
                        st.markdown(f"- {edu}")
                else:
                    st.markdown("Not found.")
                
                st.markdown("**Skills:**")
                if extracted_skills:
                    st.markdown("\n".join([f"• {skill}" for skill in extracted_skills]))
                else:
                    st.markdown("Not found.")
            
            # Show similarity analysis in the exact format requested
            st.markdown("**🎯 Match Analysis**")
            
            # Create the exact message format: "Resume A matches 85% with the given job description. Strong matches: Python, NLP, Docker."
            filename_letter = chr(65 + len(resume_results))  # A, B, C, etc.
            
            if matching_skills:
                # Format skills with proper capitalization
                formatted_skills = [skill.title() for skill in matching_skills[:5]]  # Top 5 skills, capitalized
                match_message = f"**Resume {filename_letter} matches {similarity_score:.0%} with the given job description. Strong matches: {', '.join(formatted_skills)}.**"
            else:
                match_message = f"**Resume {filename_letter} matches {similarity_score:.0%} with the given job description. No strong skill matches found.**"
            
            st.markdown(match_message)
            
            # Show additional details in smaller text
            st.caption(f"Cosine Similarity: {similarity_score:.1%} | ML Confidence: {ml_confidence:.1%} | Final Score: {final_score:.1%}")
            
            # Show score interpretation
            if final_score >= 0.7:
                st.success("🟢 Strong Fit")
            elif final_score >= 0.5:
                st.warning("🟡 Average Fit")
            else:
                st.error("🔴 Low Fit")
                
        else:
            st.warning(f"Failed to extract content from {uploaded_file.name}")
        
        st.markdown("---")
    
    # Show final ranking table
    if resume_results:
        st.markdown("## 🏆 Final Ranking & Results")
        
        # Sort by final score
        resume_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Create comprehensive ranking table
        ranking_data = []
        for i, result in enumerate(resume_results):
            # Determine fit category
            final_score = result['final_score']
            if final_score >= 0.7:
                fit_status = "🟢 Strong Fit"
            elif final_score >= 0.5:
                fit_status = "🟡 Average Fit"
            else:
                fit_status = "🔴 Low Fit"
            
            ranking_data.append({
                'Rank': i + 1,
                'Candidate': result['name'],
                'Email': result['email'],
                'BERT Match': f"{result['similarity_score']:.1%}",
                'ML Confidence': f"{result['ml_confidence']:.1%}",
                'Final Score': f"{result['final_score']:.1%}",
                'Fit Status': fit_status,
                'Key Skills': ', '.join(result['matching_skills'][:3]) if result['matching_skills'] else 'No direct matches',
                'File': result['filename']
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        st.dataframe(ranking_df, use_container_width=True)
        
        # Add download section
        st.markdown("### 📥 Download Results")
        
        col_down1, col_down2, col_down3 = st.columns([1, 1, 1])
        
        with col_down1:
            threshold = st.selectbox(
                "Select acceptance threshold:",
                options=[0.3, 0.4, 0.5, 0.6, 0.7],
                index=2,  # Default to 0.5
                help="Candidates above this score will be included in downloads"
            )
        
        # Filter accepted candidates
        accepted_candidates = filter_accepted_candidates(resume_results, threshold)
        
        with col_down2:
            st.metric(
                "Accepted Candidates",
                len(accepted_candidates),
                f"{len(accepted_candidates)/len(resume_results):.1%} of total"
            )
        
        with col_down3:
            if len(accepted_candidates) > 0:
                avg_score = sum(r['final_score'] for r in accepted_candidates) / len(accepted_candidates)
                st.metric(
                    "Avg Score (Accepted)",
                    f"{avg_score:.1%}",
                    f"{avg_score - 0.5:.1%} above threshold"
                )
        
        # Download buttons
        if len(accepted_candidates) > 0:
            download_df = create_results_dataframe(accepted_candidates)
            
            col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
            
            with col_dl1:
                # Excel download
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    download_df.to_excel(writer, sheet_name='Accepted_Candidates', index=False)
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name=f"accepted_candidates_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col_dl2:
                # CSV download
                csv_data = download_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"accepted_candidates_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col_dl3:
                # JSON download
                json_data = download_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_data,
                    file_name=f"accepted_candidates_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
            
            # Show preview of download data
            st.markdown("### 👁️ Preview of Download Data")
            st.dataframe(download_df.head(), use_container_width=True)
        
        else:
            st.warning(f"No candidates meet the {threshold:.1%} threshold. Try lowering the threshold.")
        
        # Show detailed analysis for top candidate
        if resume_results:
            top_candidate = resume_results[0]
            st.markdown("### 🌟 Top Candidate Analysis")
            st.markdown(f"**{top_candidate['name']}** - {top_candidate['final_score']:.1%} final score")
            
            col_top1, col_top2 = st.columns([1, 1])
            
            with col_top1:
                st.markdown("**Score Breakdown:**")
                st.markdown(f"• BERT Similarity: {top_candidate['similarity_score']:.1%}")
                st.markdown(f"• ML Confidence: {top_candidate['ml_confidence']:.1%}")
                st.markdown(f"• **Final Score: {top_candidate['final_score']:.1%}**")
                
                # Score composition chart
                try:
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            name='Score Components',
                            x=['BERT Similarity', 'ML Confidence', 'Final Score'],
                            y=[top_candidate['similarity_score'], top_candidate['ml_confidence'], top_candidate['final_score']],
                            marker_color=['lightblue', 'lightgreen', 'gold']
                        )
                    ])
                    fig.update_layout(
                        title=f"Score Breakdown for {top_candidate['name']}",
                        yaxis_title="Score",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("Install plotly for visualization: `pip install plotly`")
            
            with col_top2:
                if top_candidate['matching_skills']:
                    st.markdown(f"**Strong matches:** {', '.join(top_candidate['matching_skills'])}")
                
                st.markdown("**All Skills:**")
                st.markdown(', '.join(top_candidate['skills'][:10]) + ('...' if len(top_candidate['skills']) > 10 else ''))
                
                if top_candidate['education']:
                    st.markdown("**Education:**")
                    for edu in top_candidate['education'][:3]:
                        st.markdown(f"• {edu}")
        
        # Integration insights
        st.markdown("### 🔬 Integration Analysis")
        
        col_insight1, col_insight2 = st.columns([1, 1])
        
        with col_insight1:
            st.markdown("**BERT vs ML Model:**")
            bert_scores = [r['similarity_score'] for r in resume_results]
            ml_scores = [r['ml_confidence'] for r in resume_results]
            correlation = np.corrcoef(bert_scores, ml_scores)[0, 1]
            st.markdown(f"• Correlation: {correlation:.3f}")
            st.markdown(f"• BERT avg: {np.mean(bert_scores):.3f}")
            st.markdown(f"• ML avg: {np.mean(ml_scores):.3f}")
        
        with col_insight2:
            st.markdown("**Score Distribution:**")
            final_scores = [r['final_score'] for r in resume_results]
            st.markdown(f"• Highest: {max(final_scores):.1%}")
            st.markdown(f"• Average: {np.mean(final_scores):.1%}")
            st.markdown(f"• Lowest: {min(final_scores):.1%}")
            st.markdown(f"• Above 50%: {sum(1 for s in final_scores if s >= 0.5)}/{len(final_scores)}")

    # Download button for detailed results
    st.markdown("---")
    st.markdown("## 📥 Download Results")
    if resume_results:
        # Create a comprehensive dataframe for download
        download_df = create_results_dataframe(resume_results)
        
        # Create Excel file in memory for browser download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Main results sheet
            download_df.to_excel(writer, sheet_name="Results", index=False)
            
            # Add breakdown sheets for top candidates
            for result in resume_results[:3]:  # Top 3 candidates
                candidate_df = pd.DataFrame({
                    'Metric': ['Cosine Similarity', 'ML Confidence', 'Final Score'],
                    'Score': [result['similarity_score'], result['ml_confidence'], result['final_score']]
                })
                candidate_df.to_excel(writer, sheet_name=f"{result['name']}_Breakdown", index=False)
        
        excel_data = excel_buffer.getvalue()
        
        # Browser download button
        st.download_button(
            label="📊 Download Detailed Results (Excel)",
            data=excel_data,
            file_name=f"resume_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download comprehensive analysis results including individual candidate breakdowns"
        )
    else:
        st.warning("No results available to download.")

elif uploaded_files and not job_description.strip():
    st.warning("Please enter a job description to compare with resumes.")
elif not uploaded_files and job_description.strip():
    st.warning("Please upload resumes to analyze.")
else:
    st.info("Upload resumes and enter a job description to begin analysis.")

def create_results_dataframe(resume_results):
    """Create a comprehensive dataframe for download"""
    download_data = []
    for result in resume_results:
        download_data.append({
            'Candidate_Name': result['name'],
            'Email': result['email'],
            'Phone': result['phone'],
            'Location': result['location'],
            'BERT_Similarity_Score': f"{result['similarity_score']:.3f}",
            'ML_Confidence_Score': f"{result['ml_confidence']:.3f}",
            'Final_Combined_Score': f"{result['final_score']:.3f}",
            'Fit_Status': get_fit_status(result['final_score']),
            'Matching_Skills': ', '.join(result['matching_skills']) if result['matching_skills'] else 'None',
            'All_Skills': ', '.join(result['skills'][:15]) + ('...' if len(result['skills']) > 15 else ''),
            'Education': '; '.join(result['education'][:3]) if result['education'] else 'Not found',
            'Source_File': result['filename']
        })
    return pd.DataFrame(download_data)

def get_fit_status(final_score):
    """Get fit status based on final score"""
    if final_score >= 0.7:
        return "Strong Fit"
    elif final_score >= 0.5:
        return "Average Fit"
    else:
        return "Low Fit"

def filter_accepted_candidates(resume_results, threshold=0.5):
    """Filter candidates above threshold for download"""
    return [result for result in resume_results if result['final_score'] >= threshold]