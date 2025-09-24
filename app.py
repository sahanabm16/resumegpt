import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI-Powered ATS Resume Checker",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
import io
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import requests
import time

# Load environment variables
load_dotenv()

# Configure Google AI - Using same pattern as DOC-GPT
def configure_ai():
    """Configure Google AI with error handling"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("🔑 GOOGLE_API_KEY not found in environment variables")
            st.stop()
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"❌ Error configuring Google AI: {str(e)}")
        st.stop()
        return False

# Resume text extraction functions - Using same pattern as DOC-GPT
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

def extract_resume_text(uploaded_file):
    """Extract text from uploaded resume file"""
    if uploaded_file is not None:
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            return extract_text_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload PDF or DOCX files only.")
            return None
    return None

# AI Analysis functions - Using EXACT same pattern as DOC-GPT
def get_conversational_chain():
    """Create conversational chain with error handling using Gemini 2.0 Flash - Same as DOC-GPT"""
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3
        )
        return model
    except Exception as e:
        st.error(f"❌ Error creating conversational chain: {str(e)}")
        return None

def get_content_hash(text):
    """Generate a hash of the content to detect changes"""
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()

def is_resume_already_analyzed(resume_text):
    """Check if this resume content has already been analyzed"""
    if not resume_text or not st.session_state.last_analyzed_hash:
        return False
    
    current_hash = get_content_hash(resume_text)
    return current_hash == st.session_state.last_analyzed_hash

def analyze_resume_with_gemini(resume_text, job_description=None):
    """Analyze resume using Gemini AI - Enhanced for job-specific analysis"""
    try:
        if not resume_text.strip():
            st.error("❌ No resume text to analyze")
            return None
        
        model = get_conversational_chain()
        if not model:
            return None
        
        # Create job-specific or general analysis prompt
        if job_description and job_description.strip():
            # Job-specific analysis
            analysis_prompt = f"""
                You are an expert ATS (Applicant Tracking System) specialist. Analyze how well this resume matches the specific job requirements.

                JOB DESCRIPTION:
                {job_description}

                RESUME TO ANALYZE:
                {resume_text}

                IMPORTANT: Return ONLY a valid JSON object with the following exact structure. Do not include any text before or after the JSON:

                {{
                    "overall_score": <score_0_to_100_based_on_job_match>,
                    "ats_compatibility": <score_0_to_100>,
                    "job_match_score": <score_0_to_100>,
                    "sections_analysis": {{
                        "contact_info": {{
                            "score": <score_0_to_100>,
                            "issues": ["list of issues"],
                            "suggestions": ["list of job-specific suggestions"],
                            "content": "extracted content"
                        }},
                        "professional_summary": {{
                            "score": <score_0_to_100>,
                            "issues": ["alignment issues with job requirements"],
                            "suggestions": ["how to better match job requirements"],
                            "content": "extracted content"
                        }},
                        "work_experience": {{
                            "score": <score_0_to_100>,
                            "issues": ["missing relevant experience for this job"],
                            "suggestions": ["how to highlight relevant experience"],
                            "content": "extracted content"
                        }},
                        "education": {{
                            "score": <score_0_to_100>,
                            "issues": ["education requirements vs candidate"],
                            "suggestions": ["education positioning for this role"],
                            "content": "extracted content"
                        }},
                        "skills": {{
                            "score": <score_0_to_100>,
                            "issues": ["missing required skills from job posting"],
                            "suggestions": ["which skills to emphasize/add"],
                            "content": "extracted content"
                        }}
                    }},
                    "keywords": {{
                        "missing_keywords": ["important keywords from job posting not found in resume"],
                        "found_keywords": ["job-relevant keywords found in resume"],
                        "keyword_density": <percentage>,
                        "required_skills_missing": ["specific skills from job requirements not mentioned"],
                        "required_skills_found": ["specific skills from job requirements that are mentioned"]
                    }},
                    "formatting": {{
                        "score": <score_0_to_100>,
                        "issues": ["formatting issues"],
                        "suggestions": ["formatting suggestions"]
                    }},
                    "job_specific_analysis": {{
                        "requirements_match": <percentage_of_requirements_met>,
                        "qualification_gaps": ["list of missing qualifications"],
                        "strength_alignment": ["how candidate strengths align with job needs"],
                        "experience_relevance": <score_0_to_100>
                    }},
                    "overall_recommendations": ["top recommendations for this specific job"],
                    "ats_issues": ["ATS-specific issues for this application"],
                    "improvement_priority": ["ordered list of what to fix first for this job"]
                }}

                Focus on:
                1. How well the resume matches THIS specific job
                2. Required vs. preferred qualifications alignment
                3. Keyword optimization for THIS job posting
                4. ATS compatibility for THIS application
                5. Specific gaps to address for THIS role

                Return ONLY the JSON object, nothing else.
                """
        else:
            # General analysis (fallback)
            analysis_prompt = f"""
                You are an expert ATS (Applicant Tracking System) and HR specialist. Analyze the following resume and provide a comprehensive evaluation.

                Resume Text:
                {resume_text}

                IMPORTANT: Return ONLY a valid JSON object with the following exact structure. Do not include any text before or after the JSON:

                {{
                    "overall_score": <score_0_to_100>,
                    "ats_compatibility": <score_0_to_100>,
                    "sections_analysis": {{
                        "contact_info": {{
                            "score": <score_0_to_100>,
                            "issues": ["list of issues"],
                            "suggestions": ["list of suggestions"],
                            "content": "extracted content"
                        }},
                        "professional_summary": {{
                            "score": <score_0_to_100>,
                            "issues": ["list of issues"],
                            "suggestions": ["list of suggestions"],
                            "content": "extracted content"
                        }},
                        "work_experience": {{
                            "score": <score_0_to_100>,
                            "issues": ["list of issues"],
                            "suggestions": ["list of suggestions"],
                            "content": "extracted content"
                        }},
                        "education": {{
                            "score": <score_0_to_100>,
                            "issues": ["list of issues"],
                            "suggestions": ["list of suggestions"],
                            "content": "extracted content"
                        }},
                        "skills": {{
                            "score": <score_0_to_100>,
                            "issues": ["list of issues"],
                            "suggestions": ["list of suggestions"],
                            "content": "extracted content"
                        }}
                    }},
                    "keywords": {{
                        "missing_keywords": ["list of important missing keywords"],
                        "found_keywords": ["list of found relevant keywords"],
                        "keyword_density": <percentage>
                    }},
                    "formatting": {{
                        "score": <score_0_to_100>,
                        "issues": ["formatting issues"],
                        "suggestions": ["formatting suggestions"]
                    }},
                    "overall_recommendations": ["list of top recommendations"],
                    "ats_issues": ["list of ATS-specific issues"],
                    "improvement_priority": ["ordered list of what to fix first"]
                }}

                Focus on:
                1. ATS compatibility (keywords, formatting, structure)
                2. Content quality and relevance
                3. Professional presentation
                4. Missing information
                5. Industry standards compliance

                Return ONLY the JSON object, nothing else.
                """
        
        # Use same response handling as before
        response = model.invoke(analysis_prompt)
        
        if response and hasattr(response, 'content'):
            try:
                # Clean the response content to extract JSON
                response_text = response.content.strip()
                
                # Try to find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    analysis = json.loads(json_text)
                    return analysis
                else:
                    # If no JSON found, try parsing the whole response
                    analysis = json.loads(response_text)
                    return analysis
                    
            except json.JSONDecodeError as e:
                # If JSON parsing fails, create a structured response from text
                st.warning("⚠️ Received non-JSON response, creating structured analysis...")
                return parse_analysis_response(response.content)
        else:
            st.error("❌ Could not generate analysis")
            return None
                
    except Exception as e:
        st.error(f"❌ Error analyzing resume: {str(e)}")
        return None

def parse_analysis_response(response_text):
    """Parse non-JSON response into structured format with better text extraction"""
    
    # Try to extract meaningful content from the response
    recommendations = []
    
    # Look for common recommendation patterns in the text
    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('{') and not line.startswith('}') and len(line) > 20:
            # Skip JSON-like lines and very short lines
            if not ('score' in line and ':' in line) and not line.startswith('"'):
                recommendations.append(line)
    
    # If we couldn't extract good recommendations, provide default ones
    if not recommendations:
        recommendations = [
            "Consider adding more industry-specific keywords to improve ATS compatibility",
            "Quantify achievements with specific numbers and metrics",
            "Ensure consistent formatting throughout the resume",
            "Add a professional summary section at the top",
            "Include relevant technical skills for your industry"
        ]
    else:
        # Take only the first few meaningful recommendations
        recommendations = recommendations[:5]
    
    return {
        "overall_score": 75,
        "ats_compatibility": 70,
        "sections_analysis": {
            "contact_info": {
                "score": 80, 
                "issues": ["Missing LinkedIn profile"], 
                "suggestions": ["Add professional LinkedIn URL", "Include city and state"], 
                "content": "Contact information section"
            },
            "professional_summary": {
                "score": 75, 
                "issues": ["Generic summary"], 
                "suggestions": ["Add specific achievements", "Include target job title"], 
                "content": "Professional summary section"
            },
            "work_experience": {
                "score": 70, 
                "issues": ["Lacks quantified results"], 
                "suggestions": ["Add numbers and metrics", "Use strong action verbs"], 
                "content": "Work experience section"
            },
            "education": {
                "score": 85, 
                "issues": [], 
                "suggestions": ["Include relevant coursework", "Add GPA if above 3.5"], 
                "content": "Education section"
            },
            "skills": {
                "score": 75, 
                "issues": ["Missing technical skills"], 
                "suggestions": ["Add industry-specific skills", "Include software proficiency"], 
                "content": "Skills section"
            }
        },
        "keywords": {
            "missing_keywords": ["project management", "data analysis", "leadership", "teamwork"], 
            "found_keywords": ["communication", "problem solving"], 
            "keyword_density": 45
        },
        "formatting": {
            "score": 80, 
            "issues": ["Inconsistent spacing"], 
            "suggestions": ["Use consistent formatting", "Ensure ATS-friendly fonts"]
        },
        "overall_recommendations": recommendations,
        "ats_issues": ["Consider using standard section headings", "Avoid complex formatting"],
        "improvement_priority": ["Add more keywords", "Quantify achievements", "Improve formatting consistency"]
    }

def generate_improved_section(section_content, suggestions, job_description=None):
    """Generate improved version of a resume section - Enhanced for job-specific optimization"""
    try:
        if not section_content or not suggestions:
            return section_content
        
        is_job_specific = bool(job_description and job_description.strip())
        spinner_text = "🎯 Optimizing for this job..." if is_job_specific else "🔧 Improving section..."
        
        with st.spinner(spinner_text):
            model = get_conversational_chain()
            if not model:
                return section_content
            
            # Create job-specific or general improvement prompt
            if is_job_specific:
                improvement_prompt = f"""
                You are an expert resume writer and ATS specialist. Improve the following resume section to better match the specific job requirements.

                JOB DESCRIPTION:
                {job_description}

                ORIGINAL RESUME SECTION:
                {section_content}

                IMPROVEMENT SUGGESTIONS:
                {', '.join(suggestions)}

                Please provide an improved version that:
                1. Implements all the suggestions with job-specific focus
                2. Incorporates relevant keywords from the job description
                3. Aligns experience/skills with job requirements
                4. Uses terminology and phrases from the job posting
                5. Highlights relevant achievements for this specific role
                6. Maintains professional tone and ATS compatibility
                7. Shows clear value proposition for this position

                Focus on making this section highly relevant to the specific job requirements while maintaining authenticity.
                Return only the improved content, nothing else.
                """
            else:
                improvement_prompt = f"""
                You are an expert resume writer and ATS specialist. Please improve the following resume section based on the suggestions provided.

                Original Content:
                {section_content}

                Suggestions for improvement:
                {', '.join(suggestions)}

                Please provide an improved version that:
                1. Implements all the suggestions
                2. Maintains professional tone
                3. Is ATS-friendly with relevant keywords
                4. Uses strong action verbs and quantified achievements
                5. Follows industry best practices
                6. Is concise and impactful

                Return only the improved content, nothing else.
                """
            
            # Use same response handling as before
            response = model.invoke(improvement_prompt)
            
            if response and hasattr(response, 'content'):
                return response.content.strip()
            else:
                st.error("❌ Could not generate improvements")
                return section_content
                
    except Exception as e:
        st.error(f"❌ Error generating improvements: {str(e)}")
        return section_content

def generate_complete_resume():
    """Generate complete improved resume - Same pattern as session state handling in DOC-GPT"""
    if not st.session_state.resume_text:
        return "No resume text available"
    
    # Start with original resume
    improved_resume = st.session_state.resume_text
    
    # Replace improved sections
    for section_name, improved_content in st.session_state.improved_sections.items():
        if st.session_state.analysis_results:
            original_content = st.session_state.analysis_results.get('sections_analysis', {}).get(section_name, {}).get('content', '')
            if original_content and original_content in improved_resume:
                improved_resume = improved_resume.replace(original_content, improved_content)
    
    return improved_resume

# Main application
def main():
    """Main application function - Using same patterns as DOC-GPT"""
    # Configure AI at the start
    configure_ai()
    
    # CSS styling - Enhanced professional design
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Main Header */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        padding: 1rem 0;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        text-align: center;
        color: #4a5568;
        font-size: 1.3rem;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    
    .tagline {
        text-align: center;
        color: #718096;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
        line-height: 1.6;
    }
    
    /* Sidebar Styling */
    .sidebar-header {
        color: #2d3748;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Enhanced Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        background: linear-gradient(135deg, #5a6fd8 0%, #6b5b95 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Card Components */
    .info-box {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    
    .info-box:hover {
        transform: translateY(-2px);
    }
    
    .analysis-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #667eea;
    }
    
    .improvement-card {
        background: linear-gradient(135deg, #fef5e7 0%, #fed7aa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #f56565;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #48bb78;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced Section States */
    .fixed-section {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 50%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #48bb78;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(72, 187, 120, 0.2);
        animation: pulse-green 2s ease-in-out;
    }
    
    .fixing-section {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 50%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #ed8936;
        margin: 0.5rem 0;
        animation: pulse-orange 1.5s ease-in-out infinite;
    }
    
    /* Enhanced Animations */
    @keyframes pulse-green {
        0% { 
            box-shadow: 0 0 0 0 rgba(72, 187, 120, 0.7);
            transform: scale(1);
        }
        70% { 
            box-shadow: 0 0 0 10px rgba(72, 187, 120, 0);
            transform: scale(1.02);
        }
        100% { 
            box-shadow: 0 0 0 0 rgba(72, 187, 120, 0);
            transform: scale(1);
        }
    }
    
    @keyframes pulse-orange {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.01); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Enhanced Metrics */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* File Upload Enhancement */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px dashed #cbd5e0;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
    }
    
    /* Progress Indicators */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Enhanced Alerts */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Step Indicators */
    .step-indicator {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 1rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .metric-container {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            color: white;
        }
        
        .metric-value {
            color: white;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state - Same pattern as DOC-GPT
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = None
    if 'job_description' not in st.session_state:
        st.session_state.job_description = None
    if 'improved_sections' not in st.session_state:
        st.session_state.improved_sections = {}
    if 'response_content' not in st.session_state:
        st.session_state.response_content = None
    if 'response_type' not in st.session_state:
        st.session_state.response_type = None
    if 'response_title' not in st.session_state:
        st.session_state.response_title = None
    if 'fixed_sections' not in st.session_state:
        st.session_state.fixed_sections = set()
    if 'fixing_section' not in st.session_state:
        st.session_state.fixing_section = None
    if 'last_analyzed_hash' not in st.session_state:
        st.session_state.last_analyzed_hash = None
    if 'analysis_cached' not in st.session_state:
        st.session_state.analysis_cached = False
    
    # Enhanced Header with better visual hierarchy
    st.markdown('<h1 class="main-header">AI-Powered ATS Resume Checker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Resume Optimization with Advanced AI Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Transform Your Resume for ATS Success • Beat Applicant Tracking Systems • Land Your Dream Job</p>', unsafe_allow_html=True)
    
    # Enhanced feature highlights with consistent sizing and better descriptions
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('''
        <div class="info-box" style="min-height: 120px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎯</div>
            <b>ATS Optimize</b><br>
            <small>Beat applicant tracking systems with targeted optimization</small>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div class="info-box" style="min-height: 120px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
            <b>Smart Analysis</b><br>
            <small>AI-powered resume evaluation and scoring</small>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown('''
        <div class="info-box" style="min-height: 120px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔧</div>
            <b>Auto-Fix</b><br>
            <small>One-click resume improvements and optimization</small>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown('''
        <div class="info-box" style="min-height: 120px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚡</div>
            <b>Instant Results</b><br>
            <small>Real-time feedback and actionable insights</small>
        </div>
        ''', unsafe_allow_html=True)
    
    # st.markdown("---")
    
    # Show AI-Powered Features buttons only after analysis is complete
    if st.session_state.analysis_results:
        # Check if this is job-specific analysis
        is_job_specific = 'job_match_score' in st.session_state.analysis_results
        
        if is_job_specific:
            st.markdown("### 🎯 Job-Specific Analysis Features")
        else:
            st.markdown("### 🚀 AI-Powered Resume Features")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Analysis Overview", use_container_width=True):
                st.session_state.response_content = "analysis_complete"
                st.session_state.response_type = "analysis"
                if is_job_specific:
                    st.session_state.response_title = "🎯 Job-Specific Analysis Results"
                else:
                    st.session_state.response_title = "📊 Resume Analysis Results"
                st.rerun()
        
        with col2:
            if is_job_specific:
                button_text = "🎯 Job Match Score"
                title_text = "🎯 Job Compatibility Analysis"
            else:
                button_text = "🔍 ATS Score"
                title_text = "🔍 ATS Compatibility Score"
            
            if st.button(button_text, use_container_width=True):
                st.session_state.response_content = "ats_score"
                st.session_state.response_type = "score"
                st.session_state.response_title = title_text
                st.rerun()
        
        with col3:
            if is_job_specific:
                button_text = "💡 Job-Specific Tips"
                title_text = "💡 Job-Targeted Improvements"
            else:
                button_text = "💡 Get Suggestions"
                title_text = "💡 Improvement Suggestions"
            
            if st.button(button_text, use_container_width=True):
                st.session_state.response_content = "suggestions"
                st.session_state.response_type = "suggestions"
                st.session_state.response_title = title_text
                st.rerun()
        
        with col4:
            if is_job_specific:
                button_text = "🔧 Optimize for Job"
                title_text = "🔧 Job-Specific Optimization"
            else:
                button_text = "📝 Auto-Improve"
                title_text = "📝 Auto-Improvement Options"
            
            if st.button(button_text, use_container_width=True):
                st.session_state.response_content = "auto_improve"
                st.session_state.response_type = "improve"
                st.session_state.response_title = title_text
                st.rerun()
    
        # st.markdown("---")
    
    if st.session_state.response_content:
        display_response_content()
    else:
        # Call-to-action section moved to top for better user flow
        st.markdown('''
        <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); 
                    padding: 2rem; border-radius: 16px; text-align: center; 
                    margin: 1rem 0 2rem 0; border: 2px solid #e2e8f0;">
            <h3 style="color: #2d3748; margin-bottom: 1rem;">🚀 Ready to Optimize Your Resume?</h3>
            <p style="color: #718096; margin-bottom: 1.5rem; font-size: 1.1rem;">
                Upload your resume and job description in the sidebar to get started with AI-powered analysis!
            </p>
            <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <span style="background: #667eea; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">📄 Upload Resume</span>
                <span style="background: #48bb78; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">📋 Add Job Description</span>
                <span style="background: #ed8936; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">🎯 Get Analysis</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Enhanced visual showcase of features - moved below call-to-action
        st.markdown('<h3 style="text-align: center; color: #2d3748; margin: 2rem 0 1.5rem 0; font-weight: 600;">What You\'ll Get with Job-Specific Analysis</h3>', unsafe_allow_html=True)
        
        # Create visually appealing feature cards in a grid with consistent sizing
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('''
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 16px; color: white; 
                        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); 
                        margin-bottom: 1rem; text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
                <h4 style="margin: 0 0 0.5rem 0; color: white;">Job Match Score</h4>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">How well your resume fits the specific role</p>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); 
                        padding: 1.5rem; border-radius: 16px; color: white; 
                        box-shadow: 0 8px 25px rgba(72, 187, 120, 0.3); 
                        margin-bottom: 1rem; text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💡</div>
                <h4 style="margin: 0 0 0.5rem 0; color: white;">Targeted Suggestions</h4>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Recommendations specific to this job</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); 
                        padding: 1.5rem; border-radius: 16px; color: white; 
                        box-shadow: 0 8px 25px rgba(237, 137, 54, 0.3); 
                        margin-bottom: 1rem; text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                <h4 style="margin: 0 0 0.5rem 0; color: white;">Requirements Coverage</h4>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Percentage of job requirements you meet</p>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div style="background: linear-gradient(135deg, #9f7aea 0%, #805ad5 100%); 
                        padding: 1.5rem; border-radius: 16px; color: white; 
                        box-shadow: 0 8px 25px rgba(159, 122, 234, 0.3); 
                        margin-bottom: 1rem; text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔧</div>
                <h4 style="margin: 0 0 0.5rem 0; color: white;">Role Optimization</h4>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Tailor your resume for this position</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown('''
            <div style="background: linear-gradient(135deg, #38b2ac 0%, #319795 100%); 
                        padding: 1.5rem; border-radius: 16px; color: white; 
                        box-shadow: 0 8px 25px rgba(56, 178, 172, 0.3); 
                        margin-bottom: 1rem; text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔍</div>
                <h4 style="margin: 0 0 0.5rem 0; color: white;">Job-Specific Keywords</h4>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Keywords from the actual job posting</p>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div style="background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%); 
                        padding: 1.5rem; border-radius: 16px; color: white; 
                        box-shadow: 0 8px 25px rgba(245, 101, 101, 0.3); 
                        margin-bottom: 1rem; text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚠️</div>
                <h4 style="margin: 0 0 0.5rem 0; color: white;">Qualification Gaps</h4>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Missing requirements to address</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Sidebar for upload and analysis
    with st.sidebar:
        # st.markdown('<h2 class="sidebar-header">📄 Upload Documents</h2>', unsafe_allow_html=True)
        
        # Step 1: Resume Upload
        st.markdown("### 1️⃣ Upload Your Resume")
        uploaded_file = st.file_uploader(
            "Choose Resume File",
            type=['pdf', 'docx'],
            help="Upload your resume in PDF or DOCX format",
            key="resume_upload"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Resume uploaded: {uploaded_file.name}")
            
            # Extract text from uploaded file
            with st.spinner("📖 Extracting text from your resume..."):
                resume_text = extract_resume_text(uploaded_file)
                
            if resume_text:
                st.session_state.resume_text = resume_text
                
                # Show preview
                with st.expander("📋 Resume Text Preview"):
                    st.text_area("Extracted Text", resume_text[:500] + "..." if len(resume_text) > 500 else resume_text, height=150, disabled=True, key="resume_preview")
            else:
                st.error("❌ Failed to extract text from the resume file.")
        else:
            st.info("📤 Upload your resume to get started")
        
        st.markdown("---")
        
        # Step 2: Job Description Upload
        st.markdown("### 2️⃣ Upload Job Description")
        jd_option = st.radio(
            "How do you want to provide the job description?",
            ["Upload File (PDF/DOCX)", "Paste Text"],
            help="Choose your preferred method to provide the job description"
        )
        
        if jd_option == "Upload File (PDF/DOCX)":
            jd_file = st.file_uploader(
                "Choose Job Description File",
                type=['pdf', 'docx'],
                help="Upload the job description in PDF or DOCX format",
                key="jd_upload"
            )
            
            if jd_file is not None:
                st.success(f"✅ JD uploaded: {jd_file.name}")
                
                # Extract text from uploaded JD file
                with st.spinner("📖 Extracting text from job description..."):
                    jd_text = extract_resume_text(jd_file)  # Same function works for any text file
                    
                if jd_text:
                    st.session_state.job_description = jd_text
                    
                    # Show preview
                    with st.expander("📋 Job Description Preview"):
                        st.text_area("Extracted JD Text", jd_text[:500] + "..." if len(jd_text) > 500 else jd_text, height=150, disabled=True, key="jd_preview")
                else:
                    st.error("❌ Failed to extract text from the JD file.")
        else:
            # Text area for pasting JD
            jd_text = st.text_area(
                "Paste Job Description Here",
                height=200,
                placeholder="Paste the complete job description here...",
                help="Copy and paste the job description text"
            )
            
            if jd_text.strip():
                st.session_state.job_description = jd_text.strip()
                st.success("✅ Job description added!")
            else:
                st.info("📝 Paste the job description text above")
        
        st.markdown("---")
        
        # Step 3: Analyze Button
        st.markdown("### 3️⃣ AI Analysis")
        
        # Check if both resume and JD are available
        can_analyze = bool(st.session_state.resume_text and st.session_state.job_description)
        
        if can_analyze:
            if st.button("🤖 Analyze Resume vs Job Requirements", use_container_width=True, type="primary"):
                with st.spinner("🤖 Analyzing your resume against job requirements..."):
                    analysis = analyze_resume_with_gemini(st.session_state.resume_text, st.session_state.job_description)
                    if analysis:
                        st.session_state.analysis_results = analysis
                        st.session_state.response_content = "analysis_complete"
                        st.session_state.response_type = "analysis"
                        st.session_state.response_title = "Resume Analysis Results"
                        st.success("✅ Analysis complete! Use the buttons above to explore results →")
                        st.rerun()
        else:
            st.info("📋 Upload both resume and job description to enable analysis")
            missing_items = []
            if not st.session_state.resume_text:
                missing_items.append("• Resume")
            if not st.session_state.job_description:
                missing_items.append("• Job Description")
            st.markdown("**Missing:**")
            for item in missing_items:
                st.markdown(item)
        
        # st.markdown("---")
        # st.markdown("**ℹ️ About**")
        # st.markdown("**👨‍💻 Created by:**  Sahana")
        # st.markdown("**🤖 AI Model:** Google Gemini 2.0 Flash")
        # st.markdown("**🔒 Privacy:** Your documents are processed securely") 
        # st.markdown("**💡 Tip:** Upload both resume and job description for accurate ATS matching!")

def display_response_content():
    """Display response content based on type - Same pattern as DOC-GPT"""
    if st.session_state.response_type == "analysis":
        display_analysis_results()
    elif st.session_state.response_type == "score":
        display_ats_score()
    elif st.session_state.response_type == "suggestions":
        display_suggestions()
    elif st.session_state.response_type == "improve":
        display_auto_improve()
    elif st.session_state.response_type == "improvement":
        display_improved_section()
    
    # Clear response button
    if st.button("✖️ Clear Results", key="clear_response"):
        st.session_state.response_content = None
        st.session_state.response_type = None
        st.session_state.response_title = None
        st.rerun()

def display_analysis_results():
    """Display analysis results"""
    if not st.session_state.analysis_results:
        return
    
    analysis = st.session_state.analysis_results
    
    st.markdown(f"""
    <div style="color: #667eea; font-size: 1.8rem; font-weight: bold; margin-bottom: 1rem; text-align: center; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;">
        {st.session_state.response_title}
    </div>
    """, unsafe_allow_html=True)
    
    # Check if this is a job-specific analysis
    is_job_specific = 'job_match_score' in analysis
    
    if is_job_specific:
        # Job-specific analysis - show 4 metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score = analysis.get('overall_score', 0)
            color = get_score_color(score)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="color: {color}">{score}%</div>
                <div class="metric-label">Overall Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            job_score = analysis.get('job_match_score', 0)
            color = get_score_color(job_score)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="color: {color}">{job_score}%</div>
                <div class="metric-label">Job Match</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            ats_score = analysis.get('ats_compatibility', 0)
            color = get_score_color(ats_score)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="color: {color}">{ats_score}%</div>
                <div class="metric-label">ATS Compatibility</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            format_score = analysis.get('formatting', {}).get('score', 0)
            color = get_score_color(format_score)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="color: {color}">{format_score}%</div>
                <div class="metric-label">Formatting</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Job-specific analysis section
        if 'job_specific_analysis' in analysis:
            st.markdown("### 🎯 Job-Specific Analysis")
            job_analysis = analysis['job_specific_analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                req_match = job_analysis.get('requirements_match', 0)
                color = get_score_color(req_match)
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value" style="color: {color}">{req_match}%</div>
                    <div class="metric-label">Requirements Met</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                exp_rel = job_analysis.get('experience_relevance', 0)
                color = get_score_color(exp_rel)
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value" style="color: {color}">{exp_rel}%</div>
                    <div class="metric-label">Experience Relevance</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced visualization for qualification gaps and strengths
            qualification_gaps = job_analysis.get('qualification_gaps', [])
            strength_alignment = job_analysis.get('strength_alignment', [])
            
            if qualification_gaps or strength_alignment:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 16px; color: white; text-align: center; margin: 2rem 0 1rem 0;">
                    <h4 style="color: white; margin: 0 0 0.5rem 0;">📊 Job Compatibility Analysis</h4>
                    <p style="opacity: 0.9; margin: 0;">Your match with this specific position</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if qualification_gaps:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); 
                                    padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;">
                            <div style="text-align: center; margin-bottom: 1rem;">
                                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
                                <h4 style="color: #c53030; margin: 0;">Areas to Develop</h4>
                                <p style="color: #744210; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Focus on these for better job match</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for gap in qualification_gaps:
                            st.markdown(f"""
                            <div style="background: white; padding: 1rem; border-radius: 8px; 
                                        margin-bottom: 0.5rem; border-left: 4px solid #f56565; 
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <div style="display: flex; align-items: center;">
                                    <span style="color: #f56565; margin-right: 0.5rem;">📍</span>
                                    <span style="color: #2d3748; line-height: 1.4;">{gap}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    if strength_alignment:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); 
                                    padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;">
                            <div style="text-align: center; margin-bottom: 1rem;">
                                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💪</div>
                                <h4 style="color: #276749; margin: 0;">Your Competitive Edge</h4>
                                <p style="color: #22543d; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Strengths that match this role</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for strength in strength_alignment:
                            st.markdown(f"""
                            <div style="background: white; padding: 1rem; border-radius: 8px; 
                                        margin-bottom: 0.5rem; border-left: 4px solid #48bb78; 
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <div style="display: flex; align-items: center;">
                                    <span style="color: #48bb78; margin-right: 0.5rem;">⭐</span>
                                    <span style="color: #2d3748; line-height: 1.4;">{strength}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        
    else:
        # General analysis - show 3 metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score = analysis.get('overall_score', 0)
            color = get_score_color(score)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="color: {color}">{score}%</div>
                <div class="metric-label">Overall Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ats_score = analysis.get('ats_compatibility', 0)
            color = get_score_color(ats_score)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="color: {color}">{ats_score}%</div>
                <div class="metric-label">ATS Compatibility</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            format_score = analysis.get('formatting', {}).get('score', 0)
            color = get_score_color(format_score)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="color: {color}">{format_score}%</div>
                <div class="metric-label">Formatting Score</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced Keywords Analysis for job-specific analysis
    keywords = analysis.get('keywords', {})
    if keywords:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%); 
                    padding: 1.5rem; border-radius: 16px; color: #234e52; text-align: center; margin: 2rem 0 1rem 0;">
            <h3 style="color: #234e52; margin: 0 0 0.5rem 0;">🏷️ Keywords Analysis Overview</h3>
            <p style="opacity: 0.8; margin: 0;">Your resume's keyword performance</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Keywords Summary with visual indicators
        col1, col2 = st.columns(2)
        
        with col1:
            found_keywords = keywords.get('found_keywords', [])
            if found_keywords:
                st.markdown("""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;">
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <div style="font-size: 2rem; color: #48bb78;">✅</div>
                        <h4 style="color: #48bb78; margin: 0.5rem 0;">Found Keywords</h4>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #2d3748;">{}</div>
                    </div>
                </div>
                """.format(len(found_keywords)), unsafe_allow_html=True)
                
                st.markdown("**Top keywords in your resume:**")
                for keyword in found_keywords[:8]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); 
                                padding: 0.4rem 0.8rem; border-radius: 15px; margin: 0.2rem 0; 
                                display: inline-block; margin-right: 0.5rem;">
                        <span style="color: #22543d; font-weight: 500; font-size: 0.85rem;">✅ {keyword}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            missing_keywords = keywords.get('missing_keywords', [])
            if missing_keywords:
                st.markdown("""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;">
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <div style="font-size: 2rem; color: #f56565;">❌</div>
                        <h4 style="color: #f56565; margin: 0.5rem 0;">Missing Keywords</h4>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #2d3748;">{}</div>
                    </div>
                </div>
                """.format(len(missing_keywords)), unsafe_allow_html=True)
                
                st.markdown("**Consider adding these keywords:**")
                for keyword in missing_keywords[:8]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); 
                                padding: 0.4rem 0.8rem; border-radius: 15px; margin: 0.2rem 0; 
                                display: inline-block; margin-right: 0.5rem;">
                        <span style="color: #744210; font-weight: 500; font-size: 0.85rem;">❌ {keyword}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Job-specific keyword analysis with enhanced visuals
        if is_job_specific:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%); 
                        padding: 1.5rem; border-radius: 16px; color: white; text-align: center; margin: 2rem 0 1rem 0;">
                <h4 style="color: white; margin: 0 0 0.5rem 0;">🎯 Job-Specific Skills Analysis</h4>
                <p style="opacity: 0.9; margin: 0;">Required skills evaluation for this position</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                required_found = keywords.get('required_skills_found', [])
                if required_found:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); 
                                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <div style="font-size: 2.5rem; color: #22543d;">🎉</div>
                            <h4 style="color: #22543d; margin: 0;">Required Skills You Have</h4>
                            <p style="color: #22543d; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Great job matching the requirements!</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for skill in required_found:
                        st.markdown(f"""
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; 
                                    margin-bottom: 0.3rem; border-left: 4px solid #48bb78; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="display: flex; align-items: center;">
                                <span style="color: #48bb78; margin-right: 0.5rem;">🎯</span>
                                <span style="color: #2d3748; font-weight: 500;">{skill}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                required_missing = keywords.get('required_skills_missing', [])
                if required_missing:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); 
                                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <div style="font-size: 2.5rem; color: #c53030;">⚠️</div>
                            <h4 style="color: #c53030; margin: 0;">Missing Required Skills</h4>
                            <p style="color: #744210; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Priority areas for improvement</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for skill in required_missing:
                        st.markdown(f"""
                        <div style="background: white; padding: 0.8rem; border-radius: 8px; 
                                    margin-bottom: 0.3rem; border-left: 4px solid #f56565; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="display: flex; align-items: center;">
                                <span style="color: #f56565; margin-right: 0.5rem;">🚨</span>
                                <span style="color: #2d3748; font-weight: 500;">{skill}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Enhanced Section-wise analysis with visual cards
    st.markdown("""
    <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                padding: 1.5rem; border-radius: 16px; color: #2d3748; text-align: center; margin: 2rem 0 1rem 0;">
        <h3 style="color: #2d3748; margin: 0 0 0.5rem 0;">📋 Section-wise Analysis</h3>
        <p style="opacity: 0.8; margin: 0;">Detailed breakdown of each resume section</p>
    </div>
    """, unsafe_allow_html=True)
    
    sections = analysis.get('sections_analysis', {})
    if sections:
        # Create visual cards for each section
        section_icons = {
            'summary': '📝',
            'experience': '💼', 
            'skills': '🔧',
            'education': '🎓',
            'projects': '🚀',
            'certifications': '📜',
            'contact': '📞',
            'formatting': '🎨'
        }
        
        # Display sections in a grid
        cols = st.columns(2)
        section_items = list(sections.items())
        
        for i, (section_name, section_data) in enumerate(section_items):
            if section_data:
                with cols[i % 2]:
                    icon = section_icons.get(section_name.lower(), '📄')
                    section_title = section_name.replace('_', ' ').title()
                    
                    # Get section score if available
                    section_score = None
                    if isinstance(section_data, dict):
                        section_score = section_data.get('score', None)
                    
                    # Create expandable section card
                    if section_score is not None:
                        color = get_score_color(section_score)
                        # Create a clean card header
                        st.markdown(f"""
                        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                                    margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                                    border-left: 4px solid #667eea;">
                            <div style="text-align: right; margin-bottom: 0.5rem;">
                                <span style="background: {color}; color: white; padding: 0.3rem 0.8rem; 
                                           border-radius: 20px; font-weight: bold; font-size: 0.9rem;">{section_score}%</span>
                            </div>
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div style="display: flex; align-items: center;">
                                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                                    <h4 style="color: #2d3748; margin: 0;">{section_title}</h4>
                                </div>
                                <span style="color: #cbd5e0; font-size: 0.9rem;">Click to expand ▼</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                                    margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                                    border-left: 4px solid #667eea;">
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div style="display: flex; align-items: center;">
                                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                                    <h4 style="color: #2d3748; margin: 0;">{section_title}</h4>
                                </div>
                                <span style="color: #cbd5e0; font-size: 0.9rem;">Click to expand ▼</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Create expandable content area
                    with st.expander("📋 View Detailed Analysis", expanded=False):
                        show_section_analysis_enhanced(section_name, section_data)
    else:
        st.markdown("""
        <div style="background: #f7fafc; padding: 2rem; border-radius: 12px; text-align: center; 
                    border: 2px dashed #cbd5e0; margin: 1rem 0;">
            <div style="font-size: 2rem; margin-bottom: 1rem; opacity: 0.6;">📋</div>
            <p style="color: #718096; margin: 0;">No detailed section analysis available</p>
        </div>
        """, unsafe_allow_html=True)

def show_section_analysis_enhanced(section_name, section_data):
    """Display section analysis content with enhanced styling using proper Streamlit components"""
    if not section_data:
        st.markdown("*No analysis available*")
        return
    
    if isinstance(section_data, str):
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; margin-bottom: 0.5rem;">
            <strong style="color: #667eea;">📋 Analysis:</strong><br>
            <span style="color: #4a5568;">{section_data}</span>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if isinstance(section_data, dict):
        # Handle content/feedback first
        if 'content' in section_data and section_data['content']:
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; margin-bottom: 0.8rem;">
                <strong style="color: #667eea;">📋 Content:</strong><br>
                <span style="color: #4a5568; font-size: 0.9rem;">{section_data['content']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        if 'feedback' in section_data and section_data['feedback']:
            st.markdown(f"""
            <div style="background: #f0fff4; padding: 1rem; border-radius: 8px; border-left: 4px solid #48bb78; margin-bottom: 0.8rem;">
                <strong style="color: #48bb78;">💬 Feedback:</strong><br>
                <span style="color: #4a5568; font-size: 0.9rem;">{section_data['feedback']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Handle suggestions
        if 'suggestions' in section_data:
            suggestions = section_data['suggestions']
            if isinstance(suggestions, list) and suggestions:
                suggestions_text = "<br>".join([f"• {suggestion}" for suggestion in suggestions[:3]])
                st.markdown(f"""
                <div style="background: #fffaf0; padding: 1rem; border-radius: 8px; border-left: 4px solid #ed8936; margin-bottom: 0.8rem;">
                    <strong style="color: #ed8936;">💡 Suggestions:</strong><br>
                    <span style="color: #4a5568; font-size: 0.9rem;">{suggestions_text}</span>
                </div>
                """, unsafe_allow_html=True)
            elif isinstance(suggestions, str):
                st.markdown(f"""
                <div style="background: #fffaf0; padding: 1rem; border-radius: 8px; border-left: 4px solid #ed8936; margin-bottom: 0.8rem;">
                    <strong style="color: #ed8936;">💡 Suggestions:</strong><br>
                    <span style="color: #4a5568; font-size: 0.9rem;">{suggestions}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Handle issues
        if 'issues' in section_data:
            issues = section_data['issues']
            if isinstance(issues, list) and issues:
                issues_text = "<br>".join([f"• {issue}" for issue in issues[:3]])
                st.markdown(f"""
                <div style="background: #fef2f2; padding: 1rem; border-radius: 8px; border-left: 4px solid #f56565; margin-bottom: 0.8rem;">
                    <strong style="color: #f56565;">⚠️ Issues:</strong><br>
                    <span style="color: #4a5568; font-size: 0.9rem;">{issues_text}</span>
                </div>
                """, unsafe_allow_html=True)
            elif isinstance(issues, str):
                st.markdown(f"""
                <div style="background: #fef2f2; padding: 1rem; border-radius: 8px; border-left: 4px solid #f56565; margin-bottom: 0.8rem;">
                    <strong style="color: #f56565;">⚠️ Issues:</strong><br>
                    <span style="color: #4a5568; font-size: 0.9rem;">{issues}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Handle strengths
        if 'strengths' in section_data:
            strengths = section_data['strengths']
            if isinstance(strengths, list) and strengths:
                strengths_text = "<br>".join([f"• {strength}" for strength in strengths[:3]])
                st.markdown(f"""
                <div style="background: #f0fff4; padding: 1rem; border-radius: 8px; border-left: 4px solid #48bb78; margin-bottom: 0.8rem;">
                    <strong style="color: #48bb78;">✅ Strengths:</strong><br>
                    <span style="color: #4a5568; font-size: 0.9rem;">{strengths_text}</span>
                </div>
                """, unsafe_allow_html=True)
            elif isinstance(strengths, str):
                st.markdown(f"""
                <div style="background: #f0fff4; padding: 1rem; border-radius: 8px; border-left: 4px solid #48bb78; margin-bottom: 0.8rem;">
                    <strong style="color: #48bb78;">✅ Strengths:</strong><br>
                    <span style="color: #4a5568; font-size: 0.9rem;">{strengths}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Handle any other fields
        handled_keys = {'score', 'feedback', 'suggestions', 'issues', 'strengths', 'content'}
        for key, value in section_data.items():
            if key not in handled_keys and value:
                if isinstance(value, list):
                    items_text = "<br>".join([f"• {str(item)}" for item in value[:3]])
                    st.markdown(f"""
                    <div style="background: #f7fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #9f7aea; margin-bottom: 0.8rem;">
                        <strong style="color: #9f7aea;">📄 {key.title()}:</strong><br>
                        <span style="color: #4a5568; font-size: 0.9rem;">{items_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #f7fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #9f7aea; margin-bottom: 0.8rem;">
                        <strong style="color: #9f7aea;">📄 {key.title()}:</strong><br>
                        <span style="color: #4a5568; font-size: 0.9rem;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # If no specific content was found, show a generic message
        if not any(key in section_data for key in ['feedback', 'suggestions', 'issues', 'strengths', 'content']) and len([k for k in section_data.keys() if k != 'score']) == 0:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; color: #48bb78;">
                <span style="font-size: 1.2rem;">✅</span><br>
                <em>Analysis completed successfully</em>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; margin-bottom: 0.5rem;">
            <strong style="color: #667eea;">📋 Analysis:</strong><br>
            <span style="color: #4a5568;">{str(section_data)}</span>
        </div>
        """, unsafe_allow_html=True)

def get_section_content_for_card(section_data):
    """Format section analysis content for display inside cards with proper HTML styling"""
    if not section_data:
        return "<em>No analysis available</em>"
    
    if isinstance(section_data, str):
        return f"<div style='margin-bottom: 0.5rem;'><strong>📋 Analysis:</strong> {section_data}</div>"
    
    if isinstance(section_data, dict):
        content_parts = []
        
        # Handle content/feedback first
        if 'content' in section_data and section_data['content']:
            content_parts.append(f"""
            <div style='margin-bottom: 1rem; padding: 0.8rem; background: #f8fafc; border-radius: 6px; border-left: 3px solid #667eea;'>
                <strong style='color: #667eea;'>📋 Content:</strong><br>
                <span style='color: #4a5568; font-size: 0.9rem;'>{section_data['content']}</span>
            </div>
            """)
        
        if 'feedback' in section_data and section_data['feedback']:
            content_parts.append(f"""
            <div style='margin-bottom: 1rem; padding: 0.8rem; background: #f0fff4; border-radius: 6px; border-left: 3px solid #48bb78;'>
                <strong style='color: #48bb78;'>💬 Feedback:</strong><br>
                <span style='color: #4a5568; font-size: 0.9rem;'>{section_data['feedback']}</span>
            </div>
            """)
        
        # Handle suggestions
        if 'suggestions' in section_data:
            suggestions = section_data['suggestions']
            if isinstance(suggestions, list) and suggestions:
                suggestions_html = "<br>".join([f"• {suggestion}" for suggestion in suggestions[:3]])
                content_parts.append(f"""
                <div style='margin-bottom: 1rem; padding: 0.8rem; background: #fffaf0; border-radius: 6px; border-left: 3px solid #ed8936;'>
                    <strong style='color: #ed8936;'>💡 Suggestions:</strong><br>
                    <span style='color: #4a5568; font-size: 0.9rem;'>{suggestions_html}</span>
                </div>
                """)
            elif isinstance(suggestions, str):
                content_parts.append(f"""
                <div style='margin-bottom: 1rem; padding: 0.8rem; background: #fffaf0; border-radius: 6px; border-left: 3px solid #ed8936;'>
                    <strong style='color: #ed8936;'>💡 Suggestions:</strong><br>
                    <span style='color: #4a5568; font-size: 0.9rem;'>{suggestions}</span>
                </div>
                """)
        
        # Handle issues
        if 'issues' in section_data:
            issues = section_data['issues']
            if isinstance(issues, list) and issues:
                issues_html = "<br>".join([f"• {issue}" for issue in issues[:3]])
                content_parts.append(f"""
                <div style='margin-bottom: 1rem; padding: 0.8rem; background: #fef2f2; border-radius: 6px; border-left: 3px solid #f56565;'>
                    <strong style='color: #f56565;'>⚠️ Issues:</strong><br>
                    <span style='color: #4a5568; font-size: 0.9rem;'>{issues_html}</span>
                </div>
                """)
            elif isinstance(issues, str):
                content_parts.append(f"""
                <div style='margin-bottom: 1rem; padding: 0.8rem; background: #fef2f2; border-radius: 6px; border-left: 3px solid #f56565;'>
                    <strong style='color: #f56565;'>⚠️ Issues:</strong><br>
                    <span style='color: #4a5568; font-size: 0.9rem;'>{issues}</span>
                </div>
                """)
        
        # Handle strengths
        if 'strengths' in section_data:
            strengths = section_data['strengths']
            if isinstance(strengths, list) and strengths:
                strengths_html = "<br>".join([f"• {strength}" for strength in strengths[:3]])
                content_parts.append(f"""
                <div style='margin-bottom: 1rem; padding: 0.8rem; background: #f0fff4; border-radius: 6px; border-left: 3px solid #48bb78;'>
                    <strong style='color: #48bb78;'>✅ Strengths:</strong><br>
                    <span style='color: #4a5568; font-size: 0.9rem;'>{strengths_html}</span>
                </div>
                """)
            elif isinstance(strengths, str):
                content_parts.append(f"""
                <div style='margin-bottom: 1rem; padding: 0.8rem; background: #f0fff4; border-radius: 6px; border-left: 3px solid #48bb78;'>
                    <strong style='color: #48bb78;'>✅ Strengths:</strong><br>
                    <span style='color: #4a5568; font-size: 0.9rem;'>{strengths}</span>
                </div>
                """)
        
        # Handle any other fields
        handled_keys = {'score', 'feedback', 'suggestions', 'issues', 'strengths', 'content'}
        for key, value in section_data.items():
            if key not in handled_keys and value:
                if isinstance(value, list):
                    items_html = "<br>".join([f"• {str(item)}" for item in value[:3]])
                    content_parts.append(f"""
                    <div style='margin-bottom: 1rem; padding: 0.8rem; background: #f7fafc; border-radius: 6px; border-left: 3px solid #9f7aea;'>
                        <strong style='color: #9f7aea;'>📄 {key.title()}:</strong><br>
                        <span style='color: #4a5568; font-size: 0.9rem;'>{items_html}</span>
                    </div>
                    """)
                else:
                    content_parts.append(f"""
                    <div style='margin-bottom: 1rem; padding: 0.8rem; background: #f7fafc; border-radius: 6px; border-left: 3px solid #9f7aea;'>
                        <strong style='color: #9f7aea;'>📄 {key.title()}:</strong><br>
                        <span style='color: #4a5568; font-size: 0.9rem;'>{value}</span>
                    </div>
                    """)
        
        # If no specific content was found, show a generic message
        if not content_parts:
            return """
            <div style='text-align: center; padding: 1rem; color: #48bb78;'>
                <span style='font-size: 1.2rem;'>✅</span><br>
                <em>Analysis completed successfully</em>
            </div>
            """
        
        return "".join(content_parts)
    
    return f"<div style='margin-bottom: 0.5rem;'><strong>📋 Analysis:</strong> {str(section_data)}</div>"

def show_section_analysis_content(section_name, section_data):
    """Display section analysis content using proper markdown rendering"""
    if not section_data:
        st.markdown("*No analysis available*")
        return
    
    if isinstance(section_data, str):
        st.markdown(f"📋 {section_data}")
        return
    
    if isinstance(section_data, dict):
        # Handle different types of section data
        if 'feedback' in section_data:
            st.markdown(f"**💬 Feedback:** {section_data['feedback']}")
        
        if 'suggestions' in section_data:
            suggestions = section_data['suggestions']
            if isinstance(suggestions, list) and suggestions:
                st.markdown("**💡 Suggestions:**")
                for suggestion in suggestions[:3]:
                    st.markdown(f"• {suggestion}")
            elif isinstance(suggestions, str):
                st.markdown(f"**💡 Suggestions:** {suggestions}")
        
        if 'issues' in section_data:
            issues = section_data['issues']
            if isinstance(issues, list) and issues:
                st.markdown("**⚠️ Issues:**")
                for issue in issues[:3]:
                    st.markdown(f"• {issue}")
            elif isinstance(issues, str):
                st.markdown(f"**⚠️ Issues:** {issues}")
        
        if 'strengths' in section_data:
            strengths = section_data['strengths']
            if isinstance(strengths, list) and strengths:
                st.markdown("**✅ Strengths:**")
                for strength in strengths[:3]:
                    st.markdown(f"• {strength}")
            elif isinstance(strengths, str):
                st.markdown(f"**✅ Strengths:** {strengths}")
        
        # Handle any other fields
        handled_keys = {'score', 'feedback', 'suggestions', 'issues', 'strengths'}
        other_content = []
        for key, value in section_data.items():
            if key not in handled_keys and value:
                if isinstance(value, list):
                    other_content.append(f"**{key.title()}:**")
                    for item in value[:3]:
                        other_content.append(f"• {str(item)}")
                else:
                    other_content.append(f"**{key.title()}:** {value}")
        
        if other_content:
            for content in other_content:
                st.markdown(content)
        
        # If no specific content was found, show a generic message
        if not any(key in section_data for key in ['feedback', 'suggestions', 'issues', 'strengths']) and not other_content:
            st.markdown("✅ *Analysis completed*")
    else:
        st.markdown(f"📋 {str(section_data)}")

def show_section_analysis_inline(section_name, section_data):
    """Helper function to format section analysis for inline display"""
    if not section_data:
        return "No analysis available"
    
    if isinstance(section_data, str):
        return section_data
    
    if isinstance(section_data, dict):
        results = []
        
        # Handle different types of section data
        if 'feedback' in section_data:
            results.append(f"<strong>Feedback:</strong> {section_data['feedback']}")
        
        if 'suggestions' in section_data:
            suggestions = section_data['suggestions']
            if isinstance(suggestions, list) and suggestions:
                results.append(f"<strong>Suggestions:</strong><br>• " + "<br>• ".join(suggestions[:3]))
            elif isinstance(suggestions, str):
                results.append(f"<strong>Suggestions:</strong> {suggestions}")
        
        if 'issues' in section_data:
            issues = section_data['issues']
            if isinstance(issues, list) and issues:
                results.append(f"<strong>Issues:</strong><br>• " + "<br>• ".join(issues[:3]))
            elif isinstance(issues, str):
                results.append(f"<strong>Issues:</strong> {issues}")
        
        if 'strengths' in section_data:
            strengths = section_data['strengths']
            if isinstance(strengths, list) and strengths:
                results.append(f"<strong>Strengths:</strong><br>• " + "<br>• ".join(strengths[:3]))
            elif isinstance(strengths, str):
                results.append(f"<strong>Strengths:</strong> {strengths}")
        
        # If no specific fields, try to format general content
        if not results:
            for key, value in section_data.items():
                if key not in ['score'] and value:
                    if isinstance(value, list):
                        results.append(f"<strong>{key.title()}:</strong><br>• " + "<br>• ".join(str(v) for v in value[:3]))
                    else:
                        results.append(f"<strong>{key.title()}:</strong> {value}")
        
        return "<br><br>".join(results) if results else "Analysis completed"
    
    return str(section_data)

def display_ats_score():
    """Display ATS compatibility and job match scores"""
    if not st.session_state.analysis_results:
        return
    
    analysis = st.session_state.analysis_results
    is_job_specific = 'job_match_score' in analysis
    
    st.markdown(f"""
    <div style="color: #667eea; font-size: 1.8rem; font-weight: bold; margin-bottom: 1rem; text-align: center; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;">
        {st.session_state.response_title}
    </div>
    """, unsafe_allow_html=True)
    
    if is_job_specific:
        # Job-specific score display - show all 4 key scores
        st.markdown("### 📊 Complete Score Breakdown")
        
        # Main scores in a 2x2 grid
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall Score
            overall_score = analysis.get('overall_score', 0)
            color = get_score_color(overall_score)
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0; padding: 1rem; border: 2px solid {color}; border-radius: 10px;">
                <div style="font-size: 3rem; font-weight: bold; color: {color};">{overall_score}%</div>
                <div style="font-size: 1.1rem; color: #666;">Overall Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            # ATS Compatibility
            ats_score = analysis.get('ats_compatibility', 0)
            color = get_score_color(ats_score)
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0; padding: 1rem; border: 2px solid {color}; border-radius: 10px;">
                <div style="font-size: 3rem; font-weight: bold; color: {color};">{ats_score}%</div>
                <div style="font-size: 1.1rem; color: #666;">ATS Compatibility</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Job Match Score
            job_score = analysis.get('job_match_score', 0)
            color = get_score_color(job_score)
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0; padding: 1rem; border: 2px solid {color}; border-radius: 10px;">
                <div style="font-size: 3rem; font-weight: bold; color: {color};">{job_score}%</div>
                <div style="font-size: 1.1rem; color: #666;">Job Match Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Formatting Score
            format_score = analysis.get('formatting', {}).get('score', 0)
            color = get_score_color(format_score)
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0; padding: 1rem; border: 2px solid {color}; border-radius: 10px;">
                <div style="font-size: 3rem; font-weight: bold; color: {color};">{format_score}%</div>
                <div style="font-size: 1.1rem; color: #666;">Formatting Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Job-specific detailed metrics
        job_analysis = analysis.get('job_specific_analysis', {})
        if job_analysis:
            st.markdown("### 🎯 Job-Specific Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                req_match = job_analysis.get('requirements_match', 0)
                color = get_score_color(req_match)
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value" style="color: {color}">{req_match}%</div>
                    <div class="metric-label">Requirements Met</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                exp_rel = job_analysis.get('experience_relevance', 0)
                color = get_score_color(exp_rel)
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value" style="color: {color}">{exp_rel}%</div>
                    <div class="metric-label">Experience Relevance</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        # General ATS Score display (fallback)
        ats_score = analysis.get('ats_compatibility', 0)
        color = get_score_color(ats_score)
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <div style="font-size: 4rem; font-weight: bold; color: {color};">{ats_score}%</div>
            <div style="font-size: 1.2rem; color: #666;">ATS Compatibility Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ATS Issues (common to both types)
    ats_issues = analysis.get('ats_issues', [])
    if ats_issues:
        st.markdown("### 🚨 ATS Issues Found")
        for issue in ats_issues:
            st.markdown(f"• {issue}")
    
    # Formatting analysis
    formatting = analysis.get('formatting', {})
    if formatting.get('issues'):
        st.markdown("### 📝 Formatting Issues")
        for issue in formatting['issues']:
            st.markdown(f"• {issue}")
    
    if formatting.get('suggestions'):
        st.markdown("### 💡 Formatting Suggestions")
        for suggestion in formatting['suggestions']:
            st.markdown(f"• {suggestion}")

def display_suggestions():
    """Display improvement suggestions with enhanced visualizations and professional design"""
    if not st.session_state.analysis_results:
        return
    
    analysis = st.session_state.analysis_results
    is_job_specific = 'job_match_score' in analysis
    
    st.markdown(f"""
    <div style="color: #667eea; font-size: 1.8rem; font-weight: bold; margin-bottom: 1rem; text-align: center; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;">
        {st.session_state.response_title}
    </div>
    """, unsafe_allow_html=True)
    
    if is_job_specific:
        # Enhanced Job-specific recommendations with visual cards
        job_recommendations = analysis.get('job_specific_recommendations', [])
        if job_recommendations:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 16px; color: white; text-align: center; margin-bottom: 2rem;">
                <h3 style="color: white; margin: 0 0 1rem 0;">🎯 Job-Specific Recommendations</h3>
                <p style="opacity: 0.9; margin: 0;">Tailored advice for this specific position</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display recommendations in enhanced cards
            for i, rec in enumerate(job_recommendations, 1):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%); 
                            padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                            border-left: 4px solid #667eea; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="background: #667eea; color: white; border-radius: 50%; 
                                   width: 24px; height: 24px; display: flex; align-items: center; 
                                   justify-content: center; font-size: 0.8rem; font-weight: bold; margin-right: 1rem;">{i}</span>
                        <strong style="color: #2d3748;">Priority Recommendation</strong>
                    </div>
                    <p style="margin: 0; color: #4a5568; line-height: 1.6;">{rec}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Enhanced Qualification Gaps with visual impact
        job_analysis = analysis.get('job_specific_analysis', {})
        qualification_gaps = job_analysis.get('qualification_gaps', [])
        strength_alignment = job_analysis.get('strength_alignment', [])
        
        if qualification_gaps or strength_alignment:
            col1, col2 = st.columns(2)
            
            with col1:
                if qualification_gaps:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); 
                                padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;">
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🚫</div>
                            <h4 style="color: #c53030; margin: 0;">Missing Qualifications</h4>
                            <p style="color: #744210; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Critical gaps to address</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for gap in qualification_gaps:
                        st.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 8px; 
                                    margin-bottom: 0.5rem; border-left: 4px solid #f56565; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="display: flex; align-items: center;">
                                <span style="color: #f56565; margin-right: 0.5rem;">⚠️</span>
                                <span style="color: #2d3748;">{gap}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                if strength_alignment:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); 
                                padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;">
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">✅</div>
                            <h4 style="color: #276749; margin: 0;">Strength Alignment</h4>
                            <p style="color: #22543d; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Your advantages for this role</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for strength in strength_alignment:
                        st.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 8px; 
                                    margin-bottom: 0.5rem; border-left: 4px solid #48bb78; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="display: flex; align-items: center;">
                                <span style="color: #48bb78; margin-right: 0.5rem;">✅</span>
                                <span style="color: #2d3748;">{strength}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Enhanced Keywords Analysis with visual charts
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%); 
                    padding: 1.5rem; border-radius: 16px; color: #234e52; text-align: center; margin: 2rem 0 1rem 0;">
            <h3 style="color: #234e52; margin: 0 0 0.5rem 0;">🏷️ Keywords Analysis</h3>
            <p style="opacity: 0.8; margin: 0;">Job-specific keyword matching results</p>
        </div>
        """, unsafe_allow_html=True)
        
        keywords_data = analysis.get('keywords', {})
        
        # Create visual keyword comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Required Skills Analysis
            required_found = keywords_data.get('required_skills_found', [])
            required_missing = keywords_data.get('required_skills_missing', [])
            
            if required_found or required_missing:
                st.markdown("""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;">
                    <h4 style="color: #2d3748; margin: 0 0 1rem 0; text-align: center;">Required Skills Match</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if required_found:
                    st.markdown("**✅ Required Skills Found:**")
                    for skill in required_found[:6]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); 
                                    padding: 0.5rem 1rem; border-radius: 20px; margin: 0.3rem 0; 
                                    display: inline-block; margin-right: 0.5rem;">
                            <span style="color: #22543d; font-weight: 500;">✅ {skill}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                if required_missing:
                    st.markdown("**⚠️ Required Skills Missing:**")
                    for skill in required_missing[:6]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); 
                                    padding: 0.5rem 1rem; border-radius: 20px; margin: 0.3rem 0; 
                                    display: inline-block; margin-right: 0.5rem;">
                            <span style="color: #744210; font-weight: 500;">⚠️ {skill}</span>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            # General Keywords Analysis
            found_keywords = keywords_data.get('found_keywords', [])
            missing_keywords = keywords_data.get('missing_keywords', [])
            
            if found_keywords or missing_keywords:
                st.markdown("""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;">
                    <h4 style="color: #2d3748; margin: 0 0 1rem 0; text-align: center;">General Keywords</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if found_keywords:
                    st.markdown("**✅ Found Keywords:**")
                    for keyword in found_keywords[:6]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #bee3f8 0%, #90cdf4 100%); 
                                    padding: 0.5rem 1rem; border-radius: 20px; margin: 0.3rem 0; 
                                    display: inline-block; margin-right: 0.5rem;">
                            <span style="color: #2c5282; font-weight: 500;">✅ {keyword}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                if missing_keywords:
                    st.markdown("**❌ Missing Keywords:**")
                    for keyword in missing_keywords[:6]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #fbb6ce 0%, #f687b3 100%); 
                                    padding: 0.5rem 1rem; border-radius: 20px; margin: 0.3rem 0; 
                                    display: inline-block; margin-right: 0.5rem;">
                            <span style="color: #97266d; font-weight: 500;">❌ {keyword}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
    else:
        # General analysis suggestions with enhanced visuals
        suggestions = analysis.get('suggestions', [])
        overall_recommendations = analysis.get('overall_recommendations', [])
        
        # Combine suggestions and recommendations
        all_suggestions = suggestions + overall_recommendations
        
        if all_suggestions:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); 
                        padding: 1.5rem; border-radius: 16px; color: #8b4513; text-align: center; margin-bottom: 2rem;">
                <h3 style="color: #8b4513; margin: 0 0 0.5rem 0;">💡 Professional Enhancement Suggestions</h3>
                <p style="opacity: 0.9; margin: 0;">General recommendations to strengthen your resume</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display suggestions in categories with visual appeal
            categorized_suggestions = {
                'Content': [],
                'Formatting': [],
                'Skills': [],
                'Experience': [],
                'General': []
            }
            
            for suggestion in all_suggestions:
                # Simple categorization based on keywords
                suggestion_lower = suggestion.lower()
                if any(word in suggestion_lower for word in ['keyword', 'skill', 'technology', 'programming', 'technical']):
                    categorized_suggestions['Skills'].append(suggestion)
                elif any(word in suggestion_lower for word in ['format', 'layout', 'structure', 'organize', 'section']):
                    categorized_suggestions['Formatting'].append(suggestion)
                elif any(word in suggestion_lower for word in ['experience', 'work', 'job', 'role', 'position', 'achievement']):
                    categorized_suggestions['Experience'].append(suggestion)
                elif any(word in suggestion_lower for word in ['content', 'description', 'detail', 'information', 'summary']):
                    categorized_suggestions['Content'].append(suggestion)
                else:
                    categorized_suggestions['General'].append(suggestion)
            
            # Display categorized suggestions
            categories_with_icons = {
                'Skills': {'icon': '🔧', 'color': '#667eea', 'bg': '#e6f3ff'},
                'Content': {'icon': '📝', 'color': '#48bb78', 'bg': '#e6fffa'},
                'Experience': {'icon': '💼', 'color': '#ed8936', 'bg': '#fff5e6'},
                'Formatting': {'icon': '🎨', 'color': '#9f7aea', 'bg': '#f7fafc'},
                'General': {'icon': '💡', 'color': '#e53e3e', 'bg': '#fed7d7'}
            }
            
            cols = st.columns(2)
            col_index = 0
            
            for category, data in categorized_suggestions.items():
                if data:  # Only show categories that have suggestions
                    with cols[col_index % 2]:
                        st.markdown(f"""
                        <div style="background: {categories_with_icons[category]['bg']}; 
                                    padding: 1rem; border-radius: 12px; margin-bottom: 1rem; 
                                    border-left: 4px solid {categories_with_icons[category]['color']};">
                            <h4 style="color: {categories_with_icons[category]['color']}; margin: 0 0 1rem 0;">
                                {categories_with_icons[category]['icon']} {category} Improvements
                            </h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for suggestion in data:
                            st.markdown(f"""
                            <div style="background: white; padding: 1rem; border-radius: 8px; 
                                        margin-bottom: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <div style="display: flex; align-items: flex-start;">
                                    <span style="color: {categories_with_icons[category]['color']}; 
                                               margin-right: 0.5rem; margin-top: 0.1rem;">•</span>
                                    <span style="color: #2d3748; line-height: 1.5;">{suggestion}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    col_index += 1
        
        # General keywords analysis with enhanced visuals
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%); 
                    padding: 1.5rem; border-radius: 16px; color: #234e52; text-align: center; margin: 2rem 0 1rem 0;">
            <h3 style="color: #234e52; margin: 0 0 0.5rem 0;">🏷️ Keywords Analysis</h3>
            <p style="opacity: 0.8; margin: 0;">Resume keyword optimization insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        keywords_data = analysis.get('keywords', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            missing_keywords = keywords_data.get('missing_keywords', [])
            if missing_keywords:
                st.markdown("""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;">
                    <h4 style="color: #2d3748; margin: 0 0 1rem 0; text-align: center;">Missing Keywords</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**❌ Consider adding these keywords:**")
                for keyword in missing_keywords[:8]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #fbb6ce 0%, #f687b3 100%); 
                                padding: 0.5rem 1rem; border-radius: 20px; margin: 0.3rem 0; 
                                display: inline-block; margin-right: 0.5rem;">
                        <span style="color: #97266d; font-weight: 500;">❌ {keyword}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            found_keywords = keywords_data.get('found_keywords', [])
            if found_keywords:
                st.markdown("""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;">
                    <h4 style="color: #2d3748; margin: 0 0 1rem 0; text-align: center;">Found Keywords</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**✅ Keywords already in your resume:**")
                for keyword in found_keywords[:8]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #bee3f8 0%, #90cdf4 100%); 
                                padding: 0.5rem 1rem; border-radius: 20px; margin: 0.3rem 0; 
                                display: inline-block; margin-right: 0.5rem;">
                        <span style="color: #2c5282; font-weight: 500;">✅ {keyword}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Priority improvements (common to both types) with enhanced styling
    improvement_priority = analysis.get('improvement_priority', [])
    if improvement_priority:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                    padding: 1.5rem; border-radius: 16px; color: #744210; text-align: center; margin: 2rem 0 1rem 0;">
            <h3 style="color: #744210; margin: 0 0 0.5rem 0;">🚀 Priority Action Items</h3>
            <p style="opacity: 0.9; margin: 0;">Focus on these improvements first for maximum impact</p>
        </div>
        """, unsafe_allow_html=True)
        
        for i, priority in enumerate(improvement_priority, 1):
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                        border-left: 4px solid #f093fb; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="background: #f093fb; color: white; border-radius: 50%; 
                               width: 28px; height: 28px; display: flex; align-items: center; 
                               justify-content: center; font-size: 0.9rem; font-weight: bold; margin-right: 1rem;">{i}</span>
                    <strong style="color: #2d3748;">High Priority Action</strong>
                </div>
                <p style="margin: 0; color: #4a5568; line-height: 1.6; margin-left: 3rem;">{priority}</p>
            </div>
            """, unsafe_allow_html=True)

def display_auto_improve():
    """Display auto-improvement options - enhanced for job-specific optimization"""
    if not st.session_state.analysis_results:
        return
    
    analysis = st.session_state.analysis_results
    is_job_specific = 'job_match_score' in analysis
    
    st.markdown(f"""
    <div style="color: #667eea; font-size: 1.8rem; font-weight: bold; margin-bottom: 1rem; text-align: center; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;">
        {st.session_state.response_title}
    </div>
    """, unsafe_allow_html=True)
    
    if is_job_specific:
        st.info("🎯 **Job-Specific Optimization**: These improvements are tailored to the specific job you're applying for!")
        
        # Show job-specific optimization tips
        job_analysis = analysis.get('job_specific_analysis', {})
        if job_analysis.get('qualification_gaps'):
            st.warning("**Focus Areas**: Address missing qualifications and optimize keywords for this specific role.")
    else:
        st.info("💡 **General ATS Optimization**: Upload a job description for job-specific improvements!")
    
    # Show fixes summary if any sections have been fixed
    if st.session_state.fixed_sections:
        st.markdown("### ✅ Fixed Sections Summary")
        cols = st.columns(min(len(st.session_state.fixed_sections), 4))
        for i, section_name in enumerate(st.session_state.fixed_sections):
            with cols[i % 4]:
                if is_job_specific:
                    st.success(f"🎯 {section_name.replace('_', ' ').title()}")
                else:
                    st.success(f"✅ {section_name.replace('_', ' ').title()}")
        st.markdown("---")
    
    if is_job_specific:
        st.markdown("### 🎯 Select sections to optimize for this job:")
    else:
        st.markdown("### 🔧 Select sections to auto-improve:")
    
    sections = st.session_state.analysis_results.get('sections_analysis', {})
    for section_name, section_data in sections.items():
        if section_data and section_data.get('suggestions'):
            # Apply CSS class based on status
            container_class = ""
            if section_name in st.session_state.fixed_sections:
                container_class = "fixed-section"
            elif st.session_state.fixing_section == section_name:
                container_class = "fixing-section"
            
            if container_class:
                st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Check if this section has been fixed
                if section_name in st.session_state.fixed_sections:
                    if is_job_specific:
                        st.markdown(f"**{section_name.replace('_', ' ').title()}** 🎯 (Score: {section_data.get('score', 0)}%) - **OPTIMIZED FOR JOB**")
                    else:
                        st.markdown(f"**{section_name.replace('_', ' ').title()}** ✅ (Score: {section_data.get('score', 0)}%) - **FIXED**")
                else:
                    st.markdown(f"**{section_name.replace('_', ' ').title()}** (Score: {section_data.get('score', 0)}%)")
                
                # Show relevant suggestions preview
                suggestions = section_data.get('suggestions', [])
                if is_job_specific:
                    st.markdown(f"Job-specific improvements: {', '.join(suggestions[:2])}")
                else:
                    st.markdown(f"Suggestions: {', '.join(suggestions[:2])}")
            
            with col2:
                # Show different button states
                if section_name in st.session_state.fixed_sections:
                    if is_job_specific:
                        st.success("🎯 Optimized!")
                    else:
                        st.success("✅ Fixed!")
                elif st.session_state.fixing_section == section_name:
                    st.info("🔧 Optimizing...")
                else:
                    button_text = "🎯 Optimize" if is_job_specific else "🔧 Fix"
                    if st.button(button_text, key=f"fix_{section_name}"):
                        content = section_data.get('content', '')
                        suggestions = section_data.get('suggestions', [])
                        
                        # Pass job-specific context if available
                        job_description = st.session_state.get('job_description', '')
                        improved_content = generate_improved_section(content, suggestions, job_description if is_job_specific else None)
                        
                        if improved_content:
                            st.session_state.improved_sections[section_name] = improved_content
                            st.session_state.fixed_sections.add(section_name)
                            if is_job_specific:
                                st.success(f"🎯 {section_name.replace('_', ' ').title()} optimized for this job!")
                            else:
                                st.success(f"✅ {section_name.replace('_', ' ').title()} improved!")
                            st.rerun()
            
            if container_class:
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Show improved sections with enhanced styling
    if st.session_state.improved_sections:
        if is_job_specific:
            st.markdown("""
            <div class="success-card">
                <h3>🎯 Your Job-Optimized Sections</h3>
                <p>These sections have been optimized specifically for the job you're applying for. Copy the content to update your resume!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-card">
                <h3>✨ Your Improved Sections</h3>
                <p>Here are all the sections we've improved for you. Copy the content to update your resume!</p>
            </div>
            """, unsafe_allow_html=True)
        
        for section_name, improved_content in st.session_state.improved_sections.items():
            icon = "🎯" if is_job_specific else "📝"
            status = "Job-Optimized" if is_job_specific else "Improved"
            
            with st.expander(f"{icon} {status} {section_name.replace('_', ' ').title()} ✅", expanded=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    help_text = "Copy this job-optimized content to update your resume!" if is_job_specific else "Copy this improved content to update your resume!"
                    st.text_area(
                        f"{status} Content for {section_name.replace('_', ' ').title()}:", 
                        improved_content, 
                        height=200, 
                        key=f"improved_{section_name}",
                        help=help_text
                    )
                with col2:
                    st.markdown("**Actions:**")
                    if st.button("📋 Copy", key=f"copy_{section_name}", help="Copy to clipboard"):
                        st.success("✅ Copied!")
                    if st.button("📥 Download", key=f"download_{section_name}"):
                        st.download_button(
                            label="📄 Save as TXT",
                            data=improved_content,
                            file_name=f"{section_name}_improved.txt",
                            mime="text/plain",
                            key=f"dl_{section_name}"
                        )
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Complete Improved Resume", type="primary"):
                improved_resume = generate_complete_resume()
                st.download_button(
                    label="📄 Download Full Resume as Text",
                    data=improved_resume,
                    file_name="improved_resume.txt",
                    mime="text/plain"
                )
        with col2:
            if st.button("🗑️ Clear All Fixes", type="secondary"):
                st.session_state.improved_sections = {}
                st.session_state.fixed_sections = set()
                st.session_state.fixing_section = None
                st.success("✅ All fixes cleared!")
                st.rerun()

def display_improved_section():
    """Display improved section content"""
    if not st.session_state.response_content or not st.session_state.response_title:
        st.error("❌ No improved content to display")
        return
    
    st.markdown(f"""
    <div style="color: #48bb78; font-size: 1.8rem; font-weight: bold; margin-bottom: 1rem; text-align: center; border-bottom: 2px solid #48bb78; padding-bottom: 0.5rem;">
        ✅ {st.session_state.response_title}
    </div>
    """, unsafe_allow_html=True)
    
    # Display the improved content in a nice formatted box
    st.markdown("""
    <div class="success-card">
        <h4>📝 Improved Content</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Show the improved content
    st.text_area(
        "Improved Section Content:",
        value=st.session_state.response_content,
        height=300,
        help="This is your AI-improved section content. Copy this to update your resume!"
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy to Clipboard", key="copy_improved"):
            st.success("✅ Content copied! Paste it into your resume.")
    with col2:
        if st.button("📥 Download Section", key="download_section"):
            st.download_button(
                label="📄 Download as Text",
                data=st.session_state.response_content,
                file_name=f"{st.session_state.response_title.lower().replace(' ', '_')}.txt",
                mime="text/plain"
            )

def show_section_analysis(section_name, section_data):
    """Display analysis for a specific section - Same style as DOC-GPT"""
    score = section_data.get('score', 0)
    issues = section_data.get('issues', [])
    suggestions = section_data.get('suggestions', [])
    content = section_data.get('content', '')
    
    # Create expandable section
    with st.expander(f"📝 {section_name.replace('_', ' ').title()} (Score: {score}%)"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if content:
                st.markdown("**Current Content:**")
                st.text_area("", content, height=100, key=f"content_{section_name}")
        
        with col2:
            # Score visualization
            color = get_score_color(score)
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="width: 80px; height: 80px; border-radius: 50%; background: {color}; color: white; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: bold; margin: 0 auto;">
                    {score}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Issues and suggestions
        if issues:
            st.markdown("**🚨 Issues Found:**")
            for issue in issues:
                st.markdown(f"• {issue}")
        
        if suggestions:
            st.markdown("**💡 Suggestions:**")
            for suggestion in suggestions:
                st.markdown(f"• {suggestion}")
            
            # Auto-fix button with enhanced feedback
            if section_name in st.session_state.fixed_sections:
                # Already fixed - show persistent success
                st.success("✅ Fixed!")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔄 Fix Again", key=f"refix_{section_name}"):
                        st.session_state.fixed_sections.discard(section_name)
                        st.rerun()
                with col2:
                    if st.button(f"👁️ View Fixes", key=f"view_{section_name}"):
                        # Navigate to Auto Improve section to see all improvements
                        st.session_state.response_type = "improve"
                        st.session_state.response_title = "🔧 Auto-Improve Your Resume"
                        st.session_state.response_content = "auto_improve"
                        st.rerun()
            elif st.session_state.fixing_section == section_name:
                # Currently fixing - show progress
                st.info("🔧 Fixing... Please wait")
            else:
                # Ready to fix
                if st.button(f"🔧 Auto-Fix {section_name.replace('_', ' ').title()}", key=f"autofix_{section_name}"):
                    st.session_state.fixing_section = section_name
                    with st.spinner(f"Improving {section_name}..."):
                        improved_content = generate_improved_section(content, suggestions)
                        if improved_content:
                            st.session_state.improved_sections[section_name] = improved_content
                            st.session_state.fixed_sections.add(section_name)
                            st.session_state.fixing_section = None
                            st.success("✅ Section improved! Check results above.")
                            st.rerun()
                        else:
                            st.session_state.fixing_section = None
                            st.error("❌ Failed to improve section. Please try again.")

def get_score_color(score):
    """Get color based on score - Same logic as before"""
    if score >= 80:
        return "#48bb78"  # Green
    elif score >= 60:
        return "#38b2ac"  # Teal
    elif score >= 40:
        return "#ed8936"  # Orange
    else:
        return "#f56565"  # Red

# Main entry point
if __name__ == "__main__":
    main()
