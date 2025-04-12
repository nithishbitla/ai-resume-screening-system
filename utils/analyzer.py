import os
import nltk
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

def extract_text(filepath):
    text = ''
    if filepath.endswith('.pdf'):
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    elif filepath.endswith('.docx'):
        import docx
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text
    return text

def analyze_and_rank(resumes):
    texts = []
    names = []
    job_desc = resumes[0]['job_desc'] if resumes else ""

    for res in resumes:
        text = extract_text(res['path'])
        texts.append(text)
        names.append(res['name'])

    documents = [job_desc] + texts
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(documents)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    return [{'name': n, 'score': round(s, 2)} for n, s in ranked]
