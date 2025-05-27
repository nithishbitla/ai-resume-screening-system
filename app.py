from flask import (
    Flask, render_template, request, redirect, session, jsonify,
    send_from_directory, make_response
)
from werkzeug.utils import secure_filename
import os
import json
import time

import firebase_admin
from firebase_admin import credentials, auth
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# Load environment variables from .env file (local dev)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')
if not app.secret_key:
    print("WARNING: Using default insecure secret key. Set FLASK_SECRET_KEY in environment!")

# Optional: secure session cookie settings (adjust for production)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False  # Set to True on HTTPS production!
)

# Load Firebase config from environment and initialize Firebase Admin SDK
firebase_creds_json = os.environ.get('FIREBASE_CONFIG_JSON')
if not firebase_creds_json:
    raise ValueError(
        "Missing FIREBASE_CONFIG_JSON environment variable. "
        "Please set it in your environment or .env file."
    )

try:
    firebase_creds = json.loads(firebase_creds_json)

    # ✅ Fix private_key formatting
    if 'private_key' in firebase_creds:
        firebase_creds['private_key'] = firebase_creds['private_key'].replace('\\n', '\n')

    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in FIREBASE_CONFIG_JSON: {e}")

except Exception as e:
    raise RuntimeError(f"Failed to initialize Firebase Admin SDK: {e}")

# Upload folder setup
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory resume storage - for production replace with DB
uploaded_resumes = []

# Dummy host credentials (replace with secure auth for production)
host_credentials = {'email': 'host@example.com', 'password': 'host_password'}

# Role to Job Description mapping
ROLE_JOB_DESCRIPTION = {
    'Data Scientist': 'Analyze large amounts of raw information to find patterns and build predictive models.',
    'Software Engineer': 'Design, develop, and maintain software applications using coding principles and agile methods.',
    'Product Manager': 'Oversee product lifecycle, gather requirements, and prioritize features based on business goals.'
}

# Load sentence transformer model once globally for performance
model = SentenceTransformer('all-MiniLM-L6-v2')


@app.route('/')
def index():
    """Login page route."""
    return render_template('login.html')


@app.route('/verify-token', methods=['POST'])
def verify_token():
    """Verify Firebase ID token sent from client and set session user."""
    try:
        data = request.get_json()
        id_token = data.get('token')
        if not id_token:
            return jsonify({'success': False, 'error': 'No token provided'}), 400

        decoded_token = auth.verify_id_token(id_token)
        session['user'] = {
            'uid': decoded_token['uid'],
            'email': decoded_token.get('email'),
            'name': decoded_token.get('name', 'Unknown')
        }
        print(f"[Login Success] {session['user']['email']}")
        return jsonify({'success': True})

    except Exception as e:
        print(f"[Token Verification Error] {e}")
        return jsonify({'success': False, 'error': str(e)}), 401


@app.route('/host-login', methods=['GET', 'POST'])
def host_login():
    """Host login route."""
    if request.method == 'POST':
        email = request.form.get('email', '')
        password = request.form.get('password', '')
        if email == host_credentials['email'] and password == host_credentials['password']:
            session['host'] = True
            return redirect('/host-dashboard')
        else:
            return render_template('host_login.html', error="Invalid credentials")
    return render_template('host_login.html')


@app.route('/host-dashboard', methods=['GET', 'POST'])
def host_dashboard():
    """Host dashboard to rank resumes."""
    if not session.get('host'):
        return redirect('/host-login')

    job_description = ''
    ranked_resumes = uploaded_resumes

    if request.method == 'POST':
        role = request.form.get('role', '').strip()
        # Use mapped job description or custom input
        job_description = ROLE_JOB_DESCRIPTION.get(role) or request.form.get('job_desc', '').strip()
        if job_description:
            ranked_resumes = rank_resumes(job_description)

    return render_template(
        'host_dashboard.html',
        resumes=ranked_resumes,
        job_description=job_description,
        roles=ROLE_JOB_DESCRIPTION
    )


@app.route('/upload')
def upload_page():
    """Page for users to upload resumes."""
    if not session.get('user'):
        return redirect('/')
    return render_template('upload.html', user=session['user'], roles=list(ROLE_JOB_DESCRIPTION.keys()))


@app.route('/upload', methods=['POST'])
def upload_resume():
    """Handle resume upload by user."""
    if not session.get('user'):
        return redirect('/')

    name = request.form.get('name', '').strip()
    role = request.form.get('role', '').strip()
    job_desc = ROLE_JOB_DESCRIPTION.get(role, '')
    resume_file = request.files.get('resume')
    email = session['user'].get('email')

    if not all([name, role, resume_file]):
        return render_template(
            'upload.html',
            user=session['user'],
            roles=list(ROLE_JOB_DESCRIPTION.keys()),
            error="Please fill all fields and select a resume file."
        )

    # Secure filename with email + timestamp to avoid clashes
    timestamp = int(time.time())
    safe_email = email.replace('@', '_at_').replace('.', '_dot_')
    filename = secure_filename(f"{name}_{safe_email}_{timestamp}_{resume_file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    resume_file.save(filepath)

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
    """Serve uploaded resume files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def rank_resumes(job_description: str):
    """
    Rank resumes by semantic similarity between job description and resume text.
    Returns resumes sorted by similarity score descending.
    """
    ranked = []
    job_embedding = model.encode(job_description, convert_to_tensor=True)

    for resume in uploaded_resumes:
        try:
            reader = PdfReader(resume['filepath'])
            text = " ".join(page.extract_text() or "" for page in reader.pages)
            resume_embedding = model.encode(text, convert_to_tensor=True)
            similarity = util.cos_sim(job_embedding, resume_embedding).item()
            ranked.append({**resume, 'score': round(similarity, 3)})
        except Exception as e:
            print(f"[PDF Processing Error] {e}")
            ranked.append({**resume, 'score': 0})

    return sorted(ranked, key=lambda x: x['score'], reverse=True)


@app.route('/download-csv')
def download_csv():
    """Download CSV file of ranked resumes (host only)."""
    if not session.get('host'):
        return redirect('/host-login')

    csv_lines = [['Name', 'Email', 'Role', 'Score', 'Filename']]
    for r in uploaded_resumes:
        score = r.get('score', 0)
        csv_lines.append([r['name'], r['email'], r['role'], score, r['file']])

    csv_content = '\n'.join([','.join(map(str, row)) for row in csv_lines])
    response = make_response(csv_content)
    response.headers['Content-Disposition'] = 'attachment; filename=ranked_resumes.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response


@app.route('/logout')
def logout():
    """Clear session and logout user."""
    session.clear()
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
