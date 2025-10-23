# Resume-analyzer-
A simple NLP-based Resume Analysis project
 Resume Analyzer (NLP + Streamlit)

This project is a simple Resume Analyzer web application built with Python and NLP. The goal is to extract useful information from resumes and evaluate the relevance of skills by using similarity-based methods. The user can upload a resume, and the application will parse and analyze it through a Streamlit interface.

Features
- Extracting name, contact information, skills, education, and experience
- Skill similarity evaluation (semantic matching)
- PDF resume support
- Simple and interactive UI (Streamlit)

Technologies Used
- Python
- spaCy
- Sentence-Transformers (SBERT)
- RapidFuzz
- PyPDF2
- Streamlit

 How to Run
1. Install the required libraries:
```bash

2. Run the application:
streamlit run 11.py

3. Open the browser and go to:
http://localhost:8501

Project Structure
project-folder/
│
├─ 11.py
├─ images/
│   └─images.png
└─ README.md


Example Output:
Extracted skills, major, experiences, education 
Matching score


Possible Future Improvements:
Adding job-resume matching
Multi-language support

Author:
Dorsa Saeedy Zade
