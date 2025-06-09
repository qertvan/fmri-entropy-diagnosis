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
            --primary-color: #3b82f6; --primary-hover: #2563eb; --bg-color: #f9fafb; --card-bg: #ffffff;
            --text-dark: #1f2937; --text-light: #4b5563; --border-color: #e5e7eb;
        }
        body { font-family: 'Roboto', sans-serif; margin: 0; padding: 40px; background: var(--bg-color); color: var(--text-dark); }
        .container { max-width: 700px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 36px; font-weight: 700; }
        .header h1 span { color: var(--primary-color); }
        .header p { font-size: 18px; color: var(--text-light); }
        .card { background: var(--card-bg); padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1); margin-bottom: 30px; }
        .upload-area { border: 2px dashed var(--border-color); padding: 40px; text-align: center; border-radius: 10px; cursor: pointer; transition: background-color 0.2s; }
        .upload-area:hover { background: #eff6ff; }
        .upload-area h3 { margin: 0 0 10px 0; font-size: 20px; font-weight: 500; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: 500; }
        .form-group select { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid var(--border-color); background-color: #ffffff; font-size: 16px; cursor: pointer; }
        .btn { background: var(--primary-color); color: white; width: 100%; padding: 14px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: 700; transition: background-color 0.2s; }
        .btn:hover { background: var(--primary-hover); }
        .processing-section, .results-section { display: none; }
        .status-text { text-align: center; font-size: 18px; font-weight: 500; margin-bottom: 15px; }
        .progress-bar { width: 100%; height: 12px; background: var(--border-color); border-radius: 6px; overflow: hidden; margin: 20px 0 10px 0; }
        .progress-fill { height: 100%; background: var(--primary-color); width: 0%; transition: width 0.5s; }
        .progress-text { text-align: center; color: var(--text-light); }
        .results-header { text-align: center; margin-bottom: 30px; }
        .results-header h4 { font-size: 20px; margin: 0; font-weight: 400; }
        .results-header span { font-size: 28px; font-weight: 700; color: var(--primary-color); }
        .results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .result-bar { background: var(--bg-color); padding: 20px; border-radius: 8px; text-align: center; }
        .result-bar .label { font-size: 16px; font-weight: 500; margin-bottom: 10px; }
        .result-bar .value { font-size: 32px; font-weight: 700; }
        #class0ProbValue { color: #10b981; } /* Generic ID for the first class (e.g., Healthy) */
        #class1ProbValue { color: #ef4444; } /* Generic ID for the second class (e.g., SCZ) */
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>🧠 Neuro<span>Scope</span></h1><p>Advanced fMRI analysis for diagnostic insights.</p></div>
        <div id="uploadCard" class="card">
            <!-- --- NEW: Enabled dropdown with values --- -->
            <div class="form-group">
                <label for="diseaseSelect">Analysis Target</label>
                <select id="diseaseSelect">
                    <option value="scz" selected>Schizophrenia vs. Healthy</option>
                    <option value="adhd">ADHD vs. Healthy</option>
                    <option value="bpd">Bipolar vs. Healthy</option>
                </select>
            </div>
            <div class="upload-area" onclick="document.getElementById('fileInput').click()"><h3>📁 Select fMRI File</h3><p>Click here to choose a preprocessed .nii.gz file</p><input type="file" id="fileInput" style="display: none;"></div>
        </div>
        <div id="processingSection" class="processing-section card"><div class="status-text" id="statusText">Initializing...</div><div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div><div class="progress-text" id="progressText">0%</div></div>
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
        let currentJobId = null;
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('uploadCard').style.display = 'none';
                document.getElementById('processingSection').style.display = 'block';
                uploadFile(file);
            }
        });

        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            // --- NEW: Send the selected disease to the backend ---
            const selectedDisease = document.getElementById('diseaseSelect').value;
            formData.append('disease', selectedDisease);

            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const data = await response.json();
                currentJobId = data.job_id;
                checkStatus();
            } catch (error) { alert('Upload failed: ' + error.message); }
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
            document.getElementById('fileInput').value = '';
            currentJobId = null;
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


# --- NEW: /upload route now accepts the selected disease ---
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        disease_key = request.form.get('disease', 'scz')  # Default to 'scz' if not provided
        if file.filename == '': return jsonify({'error': 'No file selected'}), 400

        job_id = str(uuid.uuid4())
        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        jobs[job_id] = {'status': 'preprocessing', 'progress': 0, 'filepath': filepath}

        # Pass the disease_key to the processing thread
        thread = threading.Thread(target=process_pipeline, args=(job_id, disease_key))
        thread.daemon = True
        thread.start()
        return jsonify({'job_id': job_id, 'message': 'Upload successful, processing started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- NEW: process_pipeline now accepts the disease_key ---
def process_pipeline(job_id, disease_key):
    try:
        print(f"🚀 RUNNING IN FAST CHECK MODE for disease: {disease_key.upper()} 🚀")
        preprocessed_data_dir = os.path.abspath('fast_check_data')
        jobs[job_id].update({'status': 'custom_processing', 'progress': 10})

        final_processed_file = fmri_processing.run_nilearn_processing(preprocessed_data_dir, job_id)
        jobs[job_id].update({'status': 'entropy', 'progress': 40})

        entropy_features = entropy_calculator.calculate_entropy_features(final_processed_file)
        jobs[job_id].update({'status': 'prediction', 'progress': 80})

        # Pass the disease_key to the prediction function
        results = ml_predictor.run_ml_prediction(entropy_features, disease_key)

        time.sleep(2)
        jobs[job_id].update({'status': 'completed', 'progress': 100, 'results': results})

    except Exception as e:
        import traceback
        print("\n" + "=" * 80);
        print("🚨🚨🚨 AN ERROR OCCURRED IN THE PIPELINE! 🚨🚨🚨");
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80 + "\n");
        jobs[job_id].update({'status': 'error', 'error': str(e)})


@app.route('/status/<job_id>')
def get_status(job_id):
    if job_id not in jobs: return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])


if __name__ == '__main__':
    print("🧠 NeuroScope Server Starting...")
    print("📍 Open your browser to: http://localhost:5001")
    app.run(debug=True, host='localhost', port=5001)
