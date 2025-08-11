## Resume-Parser

# 📝 Automated Resume Parsing & Matching Tool  

An intelligent **resume screening tool** that helps recruiters quickly find the best candidates.  
It parses resumes (PDF/DOCX), extracts key details (name, email, skills, experience), matches them with a **job description**, and **ranks candidates** based on their suitability.  

The tool uses:  
✅ **BERT embeddings + cosine similarity** for semantic matching  
✅ **SpaCy + Regex** for extracting key information  
✅ **Streamlit** for a simple, recruiter-friendly web interface  
✅ *(Optional)* A trained ML model to refine similarity scores using past hire/reject data  

---

## 🚀 Features  

- **Upload multiple resumes** (PDF/DOCX)  
- **Paste a job description**  
- **Parse & extract resume info**:  
  - Name, Email, Phone  
  - Skills  
  - Education & Work Experience  
- **Match resumes to job description** using semantic similarity (BERT)  
- **Rank candidates** by match percentage  
- *(Optional)* **ML refinement layer** that adjusts similarity scores based on historical recruiter decisions  
- **Clean web interface** built with Streamlit  

---

## 🛠️ Tech Stack  

- **Python 3.9+**  
- **Streamlit** – web interface  
- **PDFMiner / PyPDF2 & docx2txt** – text extraction from resumes  
- **SpaCy** – Named Entity Recognition (names, orgs, dates)  
- **Regex** – extract emails, phone numbers  
- **Sentence-BERT (from sentence-transformers)** – semantic embeddings for resumes & job descriptions  
- **Scikit-learn** – cosine similarity, optional ML model (Logistic Regression / Random Forest)  

---

## 🔄 How It Works  

1. **Upload resumes & enter job description**  
2. Tool extracts text & parses key details using SpaCy + Regex  
3. Resume & job description are converted into **BERT embeddings**  
4. **Cosine similarity** calculates how closely each resume matches the job description  
5. *(Optional)* An ML model trained on historical data refines the score  
6. Final ranked list is displayed in a clean table with match percentage & key matching skills  

---

## 📊 Example Output  

| Candidate | Email            | Match % | ML Prediction | Final Fit | Key Matching Skills |
|-----------|-----------------|---------|---------------|-----------|---------------------|
| John Doe  | john@example.com | 85%     | Likely Hire   | ✅ Good Fit | Python, NLP, Flask |
| Jane Smith| jane@example.com | 72%     | Low Hire Prob | ⚠ Average | SQL, Analytics |
| Alex Kumar| alex@example.com | 58%     | Likely Reject | ❌ Low Fit | Java, Spring |

---

## ▶️ Installation & Usage  
```
1️⃣ **Clone the repo**  
git clone https://github.com/your-username/resume-parser.git
cd resume-parser

2️⃣ Create a virtual environment & activate it
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

3️⃣ Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

4️⃣ Run the Streamlit app
streamlit run app.py
