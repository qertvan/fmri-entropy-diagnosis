# ==============================================================================
# === app.py (Revised for Multi-Disease Support) ===============================
# ==============================================================================
from flask import Flask, request, jsonify, render_template_string
import os
import time
import threading
import uuid
import numpy as np

# Import our processing modules
import fmri_processing
import entropy_calculator
import ml_predictor

app = Flask(__name__)
jobs = {}

# --- Load ALL Machine Learning Models at Startup ---
# This single line replaces the old try/except block.
ml_predictor.load_all_models()

# --- Redesigned HTML Template (with enabled dropdown) ---
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroScope - fMRI Analysis</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #4f8cff;
            --primary-hover: #2e5a8a;
            --accent-blue: #4f8cff;
            --accent-grey: #bfc7d1;
            --bg-gradient: linear-gradient(135deg, #181a1b 0%, #232526 100%);
            --card-bg: #f3f4f6;
            --text-dark: #23272f;
            --text-light: #7c7c9a;
            --border-color: #e5e7eb;
            --shadow: 0 8px 32px 0 rgba(60, 60, 60, 0.13);
        }
        body {
            font-family: 'Roboto', sans-serif;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            background: var(--bg-gradient);
            color: var(--text-dark);
        }
        .topbar {
            width: 100vw;
            display: flex;
            justify-content: flex-end;
            align-items: center;
            padding: 0 36px;
            height: 64px;
            background: rgba(30,32,34,0.10);
            position: fixed;
            top: 0;
            left: 0;
            z-index: 10;
            box-sizing: border-box;
        }
        .menu {
            display: flex;
            gap: 28px;
        }
        .menu-btn {
            background: none;
            border: none;
            color: #e5e7eb;
            font-size: 1.08rem;
            font-weight: 500;
            cursor: pointer;
            padding: 0 2px;
            transition: color 0.2s;
        }
        .menu-btn.active, .menu-btn:hover {
            color: var(--primary-color);
        }
        .container {
            max-width: 440px;
            margin: 100px auto 0 auto;
            background: #f3f6fa;
            border-radius: 20px;
            box-shadow: var(--shadow);
            padding: 44px 34px 34px 34px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .header {
            text-align: center;
            margin-bottom: 32px;
        }
        .header img {
            width: 68px;
            margin-bottom: 10px;
        }
        .header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0 0 8px 0;
            letter-spacing: -1px;
            background: linear-gradient(90deg, #4f8cff 60%, #bfc7d1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
        }
        .header h1 span {
            color: var(--primary-color);
            -webkit-text-fill-color: var(--primary-color);
            text-fill-color: var(--primary-color);
            background: none;
        }
        .header p {
            font-size: 1.13rem;
            color: #5a5e6b;
            margin: 0;
            font-weight: 400;
        }
        .form-group {
            width: 100%;
            margin-bottom: 18px;
        }
        .form-group label {
            display: block;
            margin-bottom: 7px;
            font-weight: 500;
            color: var(--text-dark);
        }
        .form-group select {
            width: 100%;
            padding: 12px 14px;
            border-radius: 10px;
            border: 1.5px solid var(--border-color);
            background-color: #f8fafc;
            font-size: 1rem;
            color: var(--text-dark);
            font-weight: 400;
            outline: none;
            transition: border 0.2s;
        }
        .form-group select:focus {
            border: 1.5px solid var(--primary-color);
        }
        .upload-area {
            width: 100%;
            background: #f3f6fa;
            border: 2px dashed var(--primary-color);
            border-radius: 16px;
            padding: 32px 0 24px 0;
            text-align: center;
            cursor: pointer;
            margin-bottom: 0;
            transition: background 0.2s, border 0.2s;
            position: relative;
        }
        .upload-area:hover {
            background: #e5e7eb;
            border-color: var(--primary-color);
        }
        .upload-area svg {
            width: 40px;
            height: 40px;
            margin-bottom: 10px;
            color: var(--primary-color);
        }
        .upload-area h3 {
            margin: 0 0 6px 0;
            font-size: 1.13rem;
            font-weight: 600;
            color: var(--primary-color);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .upload-area p {
            margin: 0;
            font-size: 1rem;
            color: var(--text-light);
        }
        .file-name {
            margin-top: 10px;
            color: var(--text-light);
            font-size: 0.98rem;
        }
        #uploadButton {
            background: linear-gradient(90deg, var(--primary-color), var(--accent-grey));
            color: white;
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 1.08rem;
            font-weight: 700;
            margin-top: 22px;
            transition: background 0.2s;
            box-shadow: 0 2px 8px 0 rgba(99,102,241,0.08);
        }
        #uploadButton:hover {
            background: linear-gradient(90deg, var(--accent-grey), var(--primary-color));
        }
        .hidden { display: none !important; }
        /* RESULTS CARD STYLES */
        .results-section {
            display: none;
            background: #f3f6fa;
            border-radius: 20px;
            box-shadow: var(--shadow);
            padding: 38px 32px 32px 32px;
            margin-top: 0;
            align-items: center;
            justify-content: center;
        }
        .results-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .results-header h4 {
            font-size: 1.25rem;
            margin: 0 0 10px 0;
            font-weight: 600;
            color: var(--primary-color);
            letter-spacing: 0.5px;
        }
        .results-header span {
            font-size: 2rem;
            font-weight: 800;
            color: var(--text-dark);
            display: block;
            margin-bottom: 10px;
        }
        .results-grid {
            display: flex;
            flex-direction: column;
            gap: 12px;
            align-items: center;
            margin-bottom: 18px;
        }
        .result-bar {
            background: #f8fafc;
            border: 1.5px solid var(--primary-color);
            border-radius: 12px;
            padding: 18px 32px;
            min-width: 220px;
            text-align: center;
            margin-bottom: 0;
        }
        .result-bar .label {
            font-size: 1.12rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--primary-color);
        }
        .result-bar .value {
            font-size: 2.1rem;
            font-weight: 800;
            color: var(--text-dark);
        }
        #class0ProbValue { color: #10b981; }
        #class1ProbValue { color: #ef4444; }
        .btn {
            background: var(--primary-color);
            color: white;
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1.08rem;
            font-weight: 700;
            margin-top: 30px;
            transition: background-color 0.2s;
        }
        .btn:hover { background: var(--primary-hover); }
        /* PROCESSING CARD STYLES */
        .processing-section {
            display: none;
            background: #f3f6fa;
            border-radius: 20px;
            box-shadow: var(--shadow);
            padding: 38px 32px 32px 32px;
            margin-top: 0;
            align-items: center;
            justify-content: center;
        }
        .status-text {
            text-align: center;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 18px;
            margin-top: 10px;
            letter-spacing: 0.2px;
        }
        .progress-bar {
            width: 100%;
            height: 14px;
            background: #e5e7eb;
            border-radius: 7px;
            overflow: hidden;
            margin: 20px 0 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--accent-grey));
            width: 0%;
            transition: width 0.5s;
        }
        .progress-text {
            text-align: center;
            color: var(--text-light);
            font-size: 1.08rem;
            font-weight: 500;
        }
        @media (max-width: 600px) {
            .container {
                max-width: 98vw;
                padding: 18px 4vw 18px 4vw;
            }
            .topbar { padding: 0 10px; }
            .results-section, .processing-section { padding: 18px 4vw 18px 4vw; }
        }
    </style>
</head>
<body>
    <div class="topbar">
        <div class="menu">
            <button class="menu-btn" id="menuHome">Homepage</button>
            <button class="menu-btn active" id="menuStart">Start Analysis</button>
            <button class="menu-btn" id="menuAbout">About</button>
        </div>
    </div>
    <div class="container" id="mainContainer">
        <div class="header">
            <img src="/static/brain-chip.png" alt="brain icon" style="width:68px; margin-bottom: 10px;" />
            <h1>Neuro<span>Scope</span></h1>
            <p>Advanced fMRI analysis for diagnostic insights.</p>
        </div>
        <div id="uploadCard">
            <div class="form-group">
                <label for="dataTypeSelect">Data Type</label>
                <select id="dataTypeSelect">
                    <option value="preprocessed" selected>Preprocessed fMRI</option>
                    <option value="raw">Raw fMRI (BIDS Dataset)</option>
                </select>
            </div>
            <div class="form-group">
                <label for="diseaseSelect">Analysis Target</label>
                <select id="diseaseSelect">
                    <option value="scz" selected>Schizophrenia vs. Healthy</option>
                    <option value="adhd">ADHD vs. Healthy</option>
                    <option value="bpd">Bipolar vs. Healthy</option>
                </select>
            </div>
            <div id="preprocessedUploadArea">
                <div class="upload-area" onclick="document.getElementById('fmriInput').click()">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 15a4 4 0 004 4h10a4 4 0 004-4M7 10l5 5m0 0l5-5m-5 5V3" /></svg>
                    <h3><span style="font-size:1.3em;">📁</span> Select fMRI File</h3>
                    <p>Choose a preprocessed <b>.nii</b> or <b>.nii.gz</b> file</p>
                    <input type="file" id="fmriInput" accept=".nii,.nii.gz,application/gzip" style="display: none;">
                    <div id="fmriFileName" class="file-name"></div>
                </div>
                <div class="upload-area" style="margin-top: 18px;" onclick="document.getElementById('tsvInput').click()">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2a2 2 0 012-2h2a2 2 0 012 2v2m-6 4h6a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                    <h3><span style="font-size:1.3em;">📊</span> Select TSV File</h3>
                    <p>Choose a <b>.tsv</b> confounds file</p>
                    <input type="file" id="tsvInput" accept=".tsv" style="display: none;">
                    <div id="tsvFileName" class="file-name"></div>
                </div>
            </div>
            <div id="rawUploadArea" style="display:none;">
                <div class="upload-area" onclick="document.getElementById('rawBidsInput').click()">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v4a1 1 0 001 1h3m10-5h3a1 1 0 011 1v4a1 1 0 01-1 1h-3m-10 4v4a1 1 0 001 1h3m10-5h3a1 1 0 011 1v4a1 1 0 01-1 1h-3m-10 4v4a1 1 0 001 1h3m10-5h3a1 1 0 011 1v4a1 1 0 01-1 1h-3" /></svg>
                    <h3><span style="font-size:1.3em;">&#128193;</span> Select BIDS Dataset</h3>
                    <p>Choose your BIDS dataset folder or select all files inside</p>
                    <input type="file" id="rawBidsInput" webkitdirectory directory multiple style="display: none;">
                    <div id="rawBidsFileName" class="file-name"></div>
                </div>
            </div>
            <button id="uploadButton" class="btn" style="margin-top: 22px; display: none;">Start Analysis</button>
        </div>
        <div id="processingSection" class="processing-section card" style="display:none;"><div class="status-text" id="statusText">Initializing...</div><div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div><div class="progress-text" id="progressText">0%</div></div>
        <div id="resultsSection" class="results-section card">
            <div class="results-header"><h4>Primary Finding</h4><span id="primaryDiagnosis"></span></div>
            <div class="results-grid">
                <div class="result-bar"><div class="label" id="class0Label">Healthy</div><div class="value" id="class0ProbValue">0%</div></div>
                <div class="result-bar"><div class="label" id="class1Label">Disease</div><div class="value" id="class1ProbValue">0%</div></div>
            </div>
            <button class="btn" onclick="resetDemo()" style="margin-top: 30px;">Analyze Another File</button>
        </div>
    </div>
    <script>
        // MENU BAR LOGIC
        function showSection(section) {
            document.getElementById('mainContainer').classList.remove('hidden');
            document.getElementById('uploadCard').style.display = (section === 'start') ? '' : 'none';
            document.getElementById('processingSection').style.display = 'none';
            document.getElementById('resultsSection').style.display = 'none';
        }
        document.getElementById('menuHome').onclick = function() {
            showSection('home');
            document.getElementById('menuHome').classList.add('active');
            document.getElementById('menuStart').classList.remove('active');
            document.getElementById('menuAbout').classList.remove('active');
            document.getElementById('uploadCard').innerHTML = '<div style="text-align:center;padding:40px 0 30px 0;"><h2 style="color:var(--primary-color);font-weight:700;">Welcome to NeuroScope</h2><p style="color:var(--text-light);font-size:1.1em;">A modern platform for advanced fMRI analysis.<br>Use the <b>Start Analysis</b> tab to begin.</p></div>';
        };
        document.getElementById('menuStart').onclick = function() {
            showSection('start');
            document.getElementById('menuHome').classList.remove('active');
            document.getElementById('menuStart').classList.add('active');
            document.getElementById('menuAbout').classList.remove('active');
            window.location.reload(); // reload to restore uploadCard
        };
        document.getElementById('menuAbout').onclick = function() {
            showSection('about');
            document.getElementById('menuHome').classList.remove('active');
            document.getElementById('menuStart').classList.remove('active');
            document.getElementById('menuAbout').classList.add('active');
            document.getElementById('uploadCard').innerHTML = '<div style="text-align:center;padding:40px 0 30px 0;"><h2 style="color:var(--primary-color);font-weight:700;">About NeuroScope</h2><p style="color:var(--text-light);font-size:1.1em;">NeuroScope is a web-based tool for fMRI data analysis, supporting both preprocessed and raw (BIDS) datasets.<br>Developed for research and clinical insights.</p></div>';
        };
        let currentJobId = null;
        let fmriFile = null;
        let tsvFile = null;
        let rawBidsFiles = [];

        document.getElementById('dataTypeSelect').addEventListener('change', function(e) {
            const type = e.target.value;
            if (type === 'preprocessed') {
                document.getElementById('preprocessedUploadArea').style.display = '';
                document.getElementById('rawUploadArea').style.display = 'none';
            } else {
                document.getElementById('preprocessedUploadArea').style.display = 'none';
                document.getElementById('rawUploadArea').style.display = '';
            }
            resetDemo();
        });

        document.getElementById('fmriInput').addEventListener('change', function(e) {
            fmriFile = e.target.files[0];
            if (fmriFile) {
                document.getElementById('fmriFileName').textContent = `Selected: ${fmriFile.name}`;
            }
            checkFilesAndShowButton();
        });

        document.getElementById('tsvInput').addEventListener('change', function(e) {
            tsvFile = e.target.files[0];
            if (tsvFile) {
                document.getElementById('tsvFileName').textContent = `Selected: ${tsvFile.name}`;
            }
            checkFilesAndShowButton();
        });

        document.getElementById('rawBidsInput').addEventListener('change', function(e) {
            rawBidsFiles = Array.from(e.target.files);
            if (rawBidsFiles.length > 0) {
                document.getElementById('rawBidsFileName').textContent = `${rawBidsFiles.length} files selected`;
            } else {
                document.getElementById('rawBidsFileName').textContent = '';
            }
            checkFilesAndShowButton();
        });

        function checkFilesAndShowButton() {
            const uploadButton = document.getElementById('uploadButton');
            const dataType = document.getElementById('dataTypeSelect').value;
            if (dataType === 'preprocessed') {
                if (fmriFile && tsvFile) {
                    uploadButton.style.display = 'block';
                } else {
                    uploadButton.style.display = 'none';
                }
            } else {
                if (rawBidsFiles.length > 0) {
                    uploadButton.style.display = 'block';
                } else {
                    uploadButton.style.display = 'none';
                }
            }
        }

        document.getElementById('uploadButton').addEventListener('click', function() {
            const dataType = document.getElementById('dataTypeSelect').value;
            if (dataType === 'preprocessed' && fmriFile && tsvFile) {
                document.getElementById('uploadCard').style.display = 'none';
                document.getElementById('processingSection').style.display = 'block';
                uploadFiles('preprocessed');
            } else if (dataType === 'raw' && rawBidsFiles.length > 0) {
                document.getElementById('uploadCard').style.display = 'none';
                document.getElementById('processingSection').style.display = 'block';
                uploadFiles('raw');
            }
        });

        async function uploadFiles(dataType) {
            const formData = new FormData();
            formData.append('data_type', dataType);
            const selectedDisease = document.getElementById('diseaseSelect').value;
            formData.append('disease', selectedDisease);
            if (dataType === 'preprocessed') {
                formData.append('fmri_file', fmriFile);
                formData.append('tsv_file', tsvFile);
            } else {
                // BIDS: send all files
                rawBidsFiles.forEach((file, idx) => {
                    formData.append('bids_files', file, file.webkitRelativePath || file.name);
                });
            }
            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.error) {
                    alert('Upload failed: ' + data.error);
                    resetDemo();
                    return;
                }
                currentJobId = data.job_id;
                checkStatus();
            } catch (error) {
                alert('Upload failed: ' + error.message);
                resetDemo();
            }
        }

        async function checkStatus() {
            if (!currentJobId) return;
            try {
                const response = await fetch(`/status/${currentJobId}`);
                const data = await response.json();
                updateProgress(data.progress, data.status);
                if (data.status === 'completed') { showResults(data.results); }
                else if (data.status === 'error') { alert('Processing failed: ' + data.error); }
                else { setTimeout(checkStatus, 1500); }
            } catch (error) { console.error('Status check failed:', error); setTimeout(checkStatus, 2000); }
        }

        function updateProgress(progress, status) {
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = Math.round(progress) + '%';
            const statusMap = { 'preprocessing': 'Analyzing fMRI data structure...','custom_processing': 'Applying advanced signal processing (NiLearn)...','entropy': 'Extracting statistical features (Entropy)...','prediction': 'Running diagnostic prediction model...','completed': 'Analysis complete!'};
            document.getElementById('statusText').textContent = statusMap[status] || 'Processing...';
        }

        // --- NEW: Dynamic results display ---
        function showResults(results) {
            document.getElementById('processingSection').style.display = 'none';
            document.getElementById('resultsSection').style.display = 'block';

            const [class0Name, class1Name] = results.class_names;
            const class0Key = class0Name.toLowerCase();
            const class1Key = class1Name.toLowerCase();

            document.getElementById('primaryDiagnosis').textContent = results.primary_diagnosis;
            document.getElementById('class0Label').textContent = class0Name;
            document.getElementById('class1Label').textContent = class1Name;
            document.getElementById('class0ProbValue').textContent = results.probabilities[class0Key].toFixed(1) + '%';
            document.getElementById('class1ProbValue').textContent = results.probabilities[class1Key].toFixed(1) + '%';
        }

        function resetDemo() {
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('uploadCard').style.display = 'block';
            document.getElementById('processingSection').style.display = 'none';
            document.getElementById('fmriInput').value = '';
            document.getElementById('tsvInput').value = '';
            document.getElementById('fmriFileName').textContent = '';
            document.getElementById('tsvFileName').textContent = '';
            document.getElementById('uploadButton').style.display = 'none';
            currentJobId = null;
            fmriFile = null;
            tsvFile = null;
            rawBidsFiles = [];
            document.getElementById('rawBidsInput').value = '';
            document.getElementById('rawBidsFileName').textContent = '';
        }
        // Hide processing/results on load
        window.onload = function() {
            document.getElementById('processingSection').style.display = 'none';
            document.getElementById('resultsSection').style.display = 'none';
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        data_type = request.form.get('data_type', 'preprocessed')
        disease_key = request.form.get('disease', 'scz')
        job_id = str(uuid.uuid4())
        os.makedirs('uploads', exist_ok=True)
        if data_type == 'preprocessed':
            if 'fmri_file' not in request.files or 'tsv_file' not in request.files:
                return jsonify({'error': 'Both fMRI and TSV files are required'}), 400
            fmri_file = request.files['fmri_file']
            tsv_file = request.files['tsv_file']
            if fmri_file.filename == '' or tsv_file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            if not (fmri_file.filename.endswith('.nii') or fmri_file.filename.endswith('.nii.gz')) or not tsv_file.filename.endswith('.tsv'):
                return jsonify({'error': 'Invalid file types. Please upload .nii/.nii.gz and .tsv files'}), 400
            fmri_filepath = os.path.join('uploads', f'{job_id}_fmri.nii.gz')
            tsv_filepath = os.path.join('uploads', f'{job_id}_tsv.tsv')
            fmri_file.save(fmri_filepath)
            tsv_file.save(tsv_filepath)
            jobs[job_id] = {
                'status': 'preprocessing',
                'progress': 0,
                'data_type': data_type,
                'fmri_filepath': fmri_filepath,
                'tsv_filepath': tsv_filepath
            }
        elif data_type == 'raw':
            # BIDS dataset: save all files to uploads/jobid/ preserving folder structure
            if 'bids_files' not in request.files:
                return jsonify({'error': 'BIDS dataset files are required'}), 400
            bids_files = request.files.getlist('bids_files')
            bids_dir = os.path.join('uploads', job_id)
            for file in bids_files:
                # file.filename is the relative path (e.g. sub-01/anat/sub-01_T1w.nii.gz)
                rel_path = file.filename
                save_path = os.path.join(bids_dir, rel_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                file.save(save_path)
            jobs[job_id] = {
                'status': 'preprocessing',
                'progress': 0,
                'data_type': data_type,
                'bids_dir': bids_dir
            }
        else:
            return jsonify({'error': 'Invalid data type'}), 400
        thread = threading.Thread(target=process_pipeline, args=(job_id, disease_key))
        thread.daemon = True
        thread.start()
        return jsonify({'job_id': job_id, 'message': 'Upload successful, processing started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_pipeline(job_id, disease_key):
    try:
        print(f"🚀 Processing files for disease: {disease_key.upper()} 🚀")
        job_info = jobs[job_id]
        data_type = job_info.get('data_type', 'preprocessed')
        if data_type == 'preprocessed':
            fmri_filepath = job_info['fmri_filepath']
            tsv_filepath = job_info['tsv_filepath']
            jobs[job_id].update({'status': 'custom_processing', 'progress': 10})
            final_processed_file = fmri_processing.run_nilearn_processing(fmri_filepath, job_id, tsv_filepath)
            jobs[job_id].update({'status': 'entropy', 'progress': 40})
            entropy_features = entropy_calculator.calculate_entropy_features(final_processed_file)
            jobs[job_id].update({'status': 'prediction', 'progress': 80})
            results = ml_predictor.run_ml_prediction(entropy_features, disease_key)
            time.sleep(2)
            jobs[job_id].update({'status': 'completed', 'progress': 100, 'results': results})
            # Clean up uploaded files
            try:
                os.remove(fmri_filepath)
                os.remove(tsv_filepath)
            except Exception as e:
                print(f"Warning: Could not clean up files: {str(e)}")
        elif data_type == 'raw':
            bids_dir = job_info['bids_dir']
            jobs[job_id].update({'status': 'fmriprep', 'progress': 10})
            # fmriprep_output should be a preprocessed file path
            fmriprep_output, tsv_output = fmri_processing.run_fmriprep(bids_dir, job_id)
            jobs[job_id].update({'status': 'custom_processing', 'progress': 40})
            final_processed_file = fmri_processing.run_nilearn_processing(fmriprep_output, job_id, tsv_output)
            jobs[job_id].update({'status': 'entropy', 'progress': 70})
            entropy_features = entropy_calculator.calculate_entropy_features(final_processed_file)
            jobs[job_id].update({'status': 'prediction', 'progress': 90})
            results = ml_predictor.run_ml_prediction(entropy_features, disease_key)
            time.sleep(2)
            jobs[job_id].update({'status': 'completed', 'progress': 100, 'results': results})
            # Clean up uploaded files
            try:
                import shutil
                shutil.rmtree(bids_dir)
                os.remove(fmriprep_output)
                os.remove(tsv_output)
            except Exception as e:
                print(f"Warning: Could not clean up files: {str(e)}")
        else:
            jobs[job_id].update({'status': 'error', 'error': 'Invalid data type'})
            return
    except Exception as e:
        import traceback
        print("\n" + "=" * 80)
        print("🚨🚨🚨 AN ERROR OCCURRED IN THE PIPELINE! 🚨🚨🚨")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80 + "\n")
        jobs[job_id].update({'status': 'error', 'error': str(e)})


@app.route('/status/<job_id>')
def get_status(job_id):
    if job_id not in jobs: return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])


if __name__ == '__main__':
    print("🧠 NeuroScope Server Starting...")
    print("📍 Open your browser to: http://localhost:5001")
    app.run(debug=True, host='localhost', port=5001)
