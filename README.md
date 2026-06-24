# Resume-Analyst 🚀

Resume-Analyst is a Flask-based web application that uses Machine Learning and AI to analyze resumes. It identifies job categories, recommends roles, extracts contact information, and provides personalized career roadmaps using Google Gemini AI.
Try yourslef :- https://huggingface.co/spaces/arunnil/Resume-Analyst

## Features ✨

- **Resume Categorization:** Uses a Hybrid Machine Learning model (Random Forest + TF-IDF) to classify resumes into various job categories.
- **Job Recommendations:** Suggests the top 5 most suitable job roles based on the resume content.
- **Information Extraction:** Automatically extracts Name, Email, Contact Number, Skills, and Education from PDF or TXT files.
- **AI Career Coaching:** Integrates **Google Gemini AI** to provide a clear career roadmap, identifying skill gaps and offering resume optimization tips.
- **Interactive Visualizations:** Displays job category probability scores using a dynamic chart.

## Tech Stack 🛠️

- **Backend:** Flask (Python)
- **Machine Learning:** Scikit-Learn (Random Forest), XGBoost, TF-IDF Vectorization.
- **AI Integration:** Google Generative AI (Gemini 1.5 Flash).
- **PDF Processing:** PyPDF2.
- **Frontend:** HTML/CSS (Jinja2 Templates).

## Installation & Setup ⚙️

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Resume-Analyst.git
   cd Resume-Analyst
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables:**
   - Create a `.env` file in the root directory.
   - Add your [Google Gemini API Key](https://aistudio.google.com/app/apikey):
     ```env
     GEMINI_API_KEY=your_api_key_here
     ```

5. **Run the application:**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

## Project Structure 📂

- `app.py`: Main Flask application containing processing logic and API integrations.
- `models/`: Contains pre-trained ML models (`.pkl` files).
- `templates/`: HTML templates for the web interface.
- `requirements.txt`: List of Python dependencies.
- `.gitignore`: Ensures sensitive files like `.env` and model files are not uploaded to GitHub.

## Usage 💡

1. Upload your resume in **PDF** or **TXT** format.
2. Click **Analyze**.
3. View your predicted category, extracted details, and AI-generated career advice!

---
*Note: This project uses pre-trained models. Ensure the `models/` folder contains the necessary pickle files for full functionality.*
