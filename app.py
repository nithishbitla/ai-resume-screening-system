from flask import Flask, render_template, request, redirect, session, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import firebase_admin
from firebase_admin import credentials, auth

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

# In-memory resume store (use DB later)
uploaded_resumes = []

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/verify-token', methods=['POST'])
def verify_token():
    try:
        data = request.get_json()
        id_token = data['token']
        decoded_token = auth.verify_id_token(id_token)
        session['user'] = {
            'uid': decoded_token['uid'],
            'email': decoded_token['email'],
            'name': decoded_token.get('name', 'Unknown')
        }
        return jsonify({'success': True})
    except Exception as e:
        print(f"[Token Error] {e}")
        return jsonify({'success': False}), 401

@app.route('/upload')
def upload_page():
    if 'user' not in session:
        return redirect('/')
    return render_template('upload.html', user=session['user'])

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'user' not in session:
        return redirect('/')

    name = request.form['name']
    role = request.form['role']
    job_desc = request.form['job_desc']
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
        'file': filename
    })

    return redirect('/dashboard')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', resumes=uploaded_resumes)

@app.route('/resume/<filename>')
def resume(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
