from flask import Flask, render_template, request, redirect, session, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename
import os
import firebase_admin
from firebase_admin import credentials, auth
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import csv

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Firebase initialization
cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred)

# Upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory storage for resumes
uploaded_resumes = []
host_credentials = {'email': 'host@example.com', 'password': 'host_password'}  # Dummy credentials

# Role-Job Description Mapping
ROLE_JOB_DESCRIPTION = {
    'Data Scientist': 'Analyze large amounts of raw information to find patterns and build predictive models.',
    'Software Engineer': 'Design, develop, and maintain software applications using coding principles and agile methods.',
    'Product Manager': 'Oversee product lifecycle, gather requirements, and prioritize features based on business goals.'
}

# Load the AI model once globally
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/verify-token', methods=['POST'])
def verify_token():
    try:
        data = request.get_json()
        id_token = data.get('token')
        if not id_token:
            raise ValueError("No token provided")

        decoded_token = auth.verify_id_token(id_token)
        session['user'] = {
            'uid': decoded_token['uid'],
            'email': decoded_token['email'],
            'name': decoded_token.get('name', 'Unknown')
        }

        print(f"[Login Success] {session['user']['email']}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"[Token Error] {e}")
        return jsonify({'success': False}), 401

@app.route('/host-login', methods=['GET', 'POST'])
def host_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == host_credentials['email'] and password == host_credentials['password']:
            session['host'] = True
            return redirect('/host-dashboard')
        else:
            return render_template('host_login.html', error="Invalid credentials")
    return render_template('host_login.html')

@app.route('/host-dashboard', methods=['GET', 'POST'])
def host_dashboard():
    if 'host' not in session:
        return redirect('/host-login')

    job_description = ''
    if request.method == 'POST':
        role = request.form['role']
        job_description = request.form['job_desc'] if not role else ROLE_JOB_DESCRIPTION.get(role, '')
        
        # Rank resumes based on the job description
        ranked_resumes = rank_resumes(job_description)
        return render_template('host_dashboard.html', resumes=ranked_resumes, 
                               job_description=job_description, roles=ROLE_JOB_DESCRIPTION)
    
    return render_template('host_dashboard.html', resumes=uploaded_resumes, roles=ROLE_JOB_DESCRIPTION)

@app.route('/upload')
def upload_page():
    if 'user' not in session:
        return redirect('/')
    return render_template('upload.html', user=session['user'], roles=list(ROLE_JOB_DESCRIPTION.keys()))


@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'user' not in session:
        return redirect('/')

    name = request.form['name']
    role = request.form['role']
    job_desc = ROLE_JOB_DESCRIPTION.get(role, '')
    resume = request.files['resume']
    email = session['user']['email']

    filename = secure_filename(f"{name}_{email.replace('@', '_')}_{resume.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    resume.save(filepath)

    uploaded_resumes.append({
        'name': name,
        'email': email,
        'role': role,
        'job_desc': job_desc,
        'file': filename,
        'filepath': filepath
    })

    return render_template('upload_success.html', user=session['user'])

@app.route('/resume/<filename>')
def resume(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# AI-powered Resume Ranking
def rank_resumes(job_description):
    ranked_resumes = []
    job_embedding = model.encode(job_description, convert_to_tensor=True)

    for resume in uploaded_resumes:
        try:
            reader = PdfReader(resume['filepath'])
            text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
            resume_embedding = model.encode(text, convert_to_tensor=True)
            similarity = util.cos_sim(job_embedding, resume_embedding).item()
            ranked_resumes.append({**resume, 'score': round(similarity, 3)})
        except Exception as e:
            print(f"[PDF Error] {e}")
            ranked_resumes.append({**resume, 'score': 0})

    return sorted(ranked_resumes, key=lambda x: x['score'], reverse=True)

@app.route('/download-csv')
def download_csv():
    if 'host' not in session:
        return redirect('/host-login')

    # Create CSV from ranked resumes
    output = []
    output.append(['Name', 'Email', 'Role', 'Score', 'Filename'])
    for r in uploaded_resumes:
        output.append([r['name'], r['email'], r['role'], r.get('score', 0), r['file']])

    # Send CSV response
    response = make_response('\n'.join([','.join(map(str, row)) for row in output]))
    response.headers['Content-Disposition'] = 'attachment; filename=ranked_resumes.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
