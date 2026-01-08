from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle
import os
from dotenv import load_dotenv
import google.generativeai as genai
import xgboost # Required for the hybrid model loading

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# --- GEMINI API CONFIGURATION ---
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# Load models
print("Loading models...")
try:
    # 1. Load the Hybrid Categorization Model
    rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
    
    # 2. Load the Categorization Vectorizer
    tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
    
    # 3. Load the Label Encoder (Decodes 0,1 -> "Data Science", "HR")
    label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))

    # 4. Load Recommendation Models
    rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
    tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    rf_classifier_categorization = None 
except Exception as e:
    print(f"An unexpected error occurred loading models: {e}")
    rf_classifier_categorization = None


# --- TEXT CLEANING ---
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def job_recommendation(resume_text):
    resume_text_cleaned = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text_cleaned])
    probs = rf_classifier_job_recommendation.predict_proba(resume_tfidf)[0]
    classes = rf_classifier_job_recommendation.classes_
    results = zip(classes, probs)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    return sorted_results

# --- UPDATED AI ADVICE FUNCTION ---
def get_ai_career_advice(target_role, current_skills):
    # Using gemini-1.5-flash (Fastest & Most Stable currently)
    model_names = ['gemini-2.5-flash']
    
    skills_str = ", ".join(current_skills) if current_skills else "General Professional Skills"
    
    # UPDATED PROMPT: Explicitly asks to state the Role Name
    prompt = f"""
    Act as an expert Career Coach and Tech Recruiter.
    Target Role: {target_role}
    Candidate's Current Skills: {skills_str}

    Task:
    1. Start your response with a clear header: "<h3>Career Roadmap for {target_role}</h3>".
    2. **Skill Gap Analysis:** Identify 3-5 specific, modern technical skills or tools this candidate is missing but NEEDS for a {target_role} position.
    3. **Resume Optimization:** Provide 3 actionable tips to make their resume sound more like a {target_role} expert (e.g., specific keywords to add, project ideas).
    
    Format: Use HTML tags (<b> for bold, <ul> <li> for lists). Keep it concise and encouraging.
    """

    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue 
            
    return "AI advice could not be generated. Please check API Key permissions."

# --- ROBUST PDF PARSING ---
def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    try:
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"PDF Parsing Error: {e}")
    return text

# --- PARSING FUNCTIONS ---
def extract_contact_number_from_resume(text):
    contact_number = None
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()
    return contact_number

def extract_email_from_resume(text):
    email = None
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()
    return email

def extract_skills_from_resume(text):
    # Using your specific skills list logic
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau', 'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib', 'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 
        'Text Mining', 'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition',
        'Recommendation Systems', 'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks',
        'Convolutional Neural Networks', 'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 
        'Decision Trees', 'Support Vector Machines', 'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 
        'Hierarchical Clustering', 'DBSCAN', 'Association Rule Learning', 'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 
        'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL', 'Big Data Analytics', 'Cloud Computing', 'Amazon Web Services (AWS)', 
        'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker', 'Kubernetes', 'Linux', 'Shell Scripting', 'Cybersecurity', 
        'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption', 'Malware Analysis', 'Digital Forensics', 'CI/CD', 
        'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration', 'Continuous Deployment', 'Software Development', 
        'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development', 'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite', 'Photoshop', 
        'Illustrator', 'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research', 
        'Customer Development', 'Lean Startup', 'Business Development', 'Sales', 'Marketing', 'Content Marketing', 
        'Social Media Marketing', 'Email Marketing', 'SEO', 'SEM', 'PPC', 'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 
        'Lead Generation', 'Customer Relationship Management (CRM)', 'Salesforce', 'HubSpot', 'Zendesk', 'Intercom', 
        'Customer Support', 'Technical Support', 'Troubleshooting', 'Ticketing Systems', 'ServiceNow', 'ITIL', 'Quality Assurance', 
        'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing', 'Performance Testing', 'Regression Testing', 
        'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing', 'Usability Testing', 'Accessibility Testing', 
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation', 'Technical Writing', 
        'Copywriting', 'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento', 
        'Shopify', 'E-commerce', 'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement', 
        'ERP Systems', 'SAP', 'Oracle', 'Microsoft Dynamics', 'Power BI', 'QlikView', 'Looker', 'Data Engineering', 'Data Governance', 
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics', 'Descriptive Analytics', 
        'Business Intelligence', 'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 
        'GraphQL', 'SOAP', 'Microservices', 'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 
        'Message Queues', 'Socket.io', 'WebSockets', 'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 
        'Zend Framework', 'ASP.NET', 'C#', 'VB.NET', 'ASP.NET MVC', 'Entity Framework', 'Spring', 'Hibernate', 'Struts', 'Kotlin', 
        'Swift', 'Objective-C', 'iOS Development', 'Android Development', 'Flutter', 'React Native', 'Ionic', 'Mobile UI/UX Design', 
        'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI', 'Falcon', 'Tornado', 'AWS Lambda', 
        'Google Cloud Functions', 'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration', 
        'Database Administration', 'MySQL', 'PostgreSQL', 'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'Cassandra', 
        'Redis', 'Elasticsearch', 'Firebase', 'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 
        'Customer Data Platforms', 'Segment', 'Salesforce Marketing Cloud', 'HubSpot CRM', 'Zapier', 'IFTTT', 'Workflow Automation', 
        'Robotic Process Automation (RPA)', 'UI Automation', 'Natural Language Generation (NLG)', 'Virtual Reality (VR)', 
        'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling', 'Animation', 'Motion Graphics', 
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya', 'Adobe After Effects', 
        'Adobe Premiere Pro', 'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing', 
        'Content Strategy', 'Conversion Rate Optimization (CRO)', 'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 
        'User Interface (UI)', 'Persona Development', 'User Journey Mapping', 'Information Architecture (IA)', 'Usability Testing', 
        'Accessibility Compliance', 'Internationalization (I18n)', 'Localization (L10n)', 'Voice User Interface (VUI)', 'Chatbots', 
        'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection', 'Sentiment Analysis', 'Image Recognition', 
        'Object Detection', 'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection', 
        'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)', 'Vulnerability Assessment', 
        'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)', 'Identity and Access Management (IAM)', 
        'Single Sign-On (SSO)', 'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)', 
        'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)', 'FastAPI', 'Hugging Face'
    ]

    found_skills = []
    if not isinstance(text, str):
        return []

    text_lower = text.lower()
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill.lower()))
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return found_skills

def extract_education_from_resume(text):
    education_list = []
    education_block_match = re.search(r"(?i)\bEDUCATION\b(.*?)\b(EXPERIENCE|PROJECTS|TECHNICAL SKILLS|CERTIFICATIONS|SKILLS)\b", text, re.DOTALL)
    
    if not education_block_match:
        return []
    
    edu_text = education_block_match.group(1)
    edu_text = re.sub(r'\s+', ' ', edu_text).strip()
    
    date_pattern = r"(?i)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s\d{4}(?:\s*–\s*Present)?|\bMarch\s\d{4}\b|\bMay\s\d{4}\b|\d{4}\s*-\s*\d{4})"
    parts = re.split(date_pattern, edu_text)
    
    if len(parts) <= 1:
        return [edu_text]
    
    for i in range(len(parts)):
        entry = parts[i].strip()
        if len(entry) > 10: 
            education_list.append(entry)
            
    return education_list if education_list else [edu_text]

def extract_name_from_resume(text):
    name = None
    first_lines = "\n".join(text.split('\n')[:10]) 
    pattern = r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})"
    match = re.search(pattern, first_lines)
    if match:
        name = match.group(1)
    return name

@app.route('/')
def resume():
    return render_template("resume.html", category_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    category_data_for_chart = None
    
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename

        if filename == '':
            return render_template("resume.html", message="No file selected.", category_data=None)

        text = '' 
        if filename.endswith('.pdf'):
            try:
                text = pdf_to_text(file)
            except Exception as e:
                return render_template('resume.html', message=f"Error processing PDF: {e}", category_data=None)
        elif filename.endswith('.txt'):
            try:
                text = file.read().decode('utf-8')
            except Exception as e:
                return render_template('resume.html', message=f"Error processing TXT: {e}", category_data=None)
        else:
            return render_template('resume.html', message="Invalid format.", category_data=None)

        if not rf_classifier_categorization:
             return render_template('resume.html', message="Model Error: Models not loaded.", category_data=None)

        # --- PREDICTION LOGIC ---
        # 1. Clean the text
        clean_text = cleanResume(text)
        
        # 2. Vectorize the FULL text (Using Hybrid Model Logic)
        text_features_cat = tfidf_vectorizer_categorization.transform([clean_text])
        
        # 3. Predict Category (Returns ID)
        prediction_id = rf_classifier_categorization.predict(text_features_cat)[0]
        
        # 4. Decode ID to Name
        predicted_category = label_encoder.inverse_transform([prediction_id])[0]

        # 5. Get Probabilities for Chart
        category_probabilities = rf_classifier_categorization.predict_proba(text_features_cat)[0]
        all_categories = label_encoder.classes_
        
        category_scores = sorted(zip(all_categories, category_probabilities), key=lambda x: x[1], reverse=True)[:6]
        category_data_for_chart = {
            'labels': [item[0] for item in category_scores],
            'scores': [item[1] * 100 for item in category_scores]
        }

        # --- EXTRACTION & RECOMMENDATION ---
        recommended_job_list = job_recommendation(text)
        phone = extract_contact_number_from_resume(text)
        email = extract_email_from_resume(text)
        extracted_skills = extract_skills_from_resume(text)
        extracted_education = extract_education_from_resume(text)
        name = extract_name_from_resume(text)
        
        # --- AI ADVICE ---
        top_job_role = recommended_job_list[0][0] if recommended_job_list else "Professional"
        ai_advice = get_ai_career_advice(top_job_role, extracted_skills)

        return render_template(
            'resume.html',
            predicted_category=predicted_category,
            recommended_job=recommended_job_list,
            phone=phone,
            name=name,
            email=email,
            extracted_skills=extracted_skills,
            extracted_education=extracted_education,
            category_data=category_data_for_chart,
            ai_advice=ai_advice
        )
    else:
        return render_template("resume.html", message="No resume uploaded.", category_data=None)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7860)