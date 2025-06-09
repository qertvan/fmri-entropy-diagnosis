# app.py - Redesigned for Binary Classification (Healthy vs. SCZ)
from flask import Flask, request, jsonify, render_template_string
import os
import time
import threading
import uuid
import numpy as np
import joblib  # Using joblib to load our trained model

# Import our processing modules
import fmri_processing
import entropy_calculator
import ml_predictor

app = Flask(__name__)

# Simple in-memory job storage
jobs = {}

ml_predictor.load_all_models()

# --- NEW: Load the Machine Learning Model ---
# Load your trained Healthy vs. SCZ model.
# Make sure this file is in the same directory as app.py
try:
    ML_MODEL = joblib.load("model_healthy_vs_scz.joblib")
    DIAGNOSES = ['Healthy', 'Schizophrenia']
    print("✅ Healthy vs. SCZ model loaded successfully.")
except FileNotFoundError:
    ML_MODEL = None
    print("🚨 WARNING: 'model_healthy_vs_scz.joblib' not found. Real predictions will fail.")

# ==============================================================================
# === NEW REDESIGNED HTML TEMPLATE =============================================
# ==============================================================================
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
            --primary-color: #3b82f6; /* Blue 500 */
            --primary-hover: #2563eb; /* Blue 600 */
            --secondary-color: #6b7280; /* Gray 500 */
            --bg-color: #f9fafb; /* Gray 50 */
            --card-bg: #ffffff;
            --text-dark: #1f2937; /* Gray 800 */
            --text-light: #4b5563; /* Gray 600 */
            --border-color: #e5e7eb; /* Gray 200 */
        }
        body {
            font-family: 'Roboto', sans-serif;
            margin: 0;
            padding: 40px;
            background: var(--bg-color);
            color: var(--text-dark);
        }
        .container {
            max-width: 700px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 36px;
            font-weight: 700;
            color: var(--text-dark);
        }
        .header h1 span {
            color: var(--primary-color);
        }
        .header p {
            font-size: 18px;
            color: var(--text-light);
        }
        .card {
            background: var(--card-bg);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed var(--border-color);
            padding: 40px;
            text-align: center;
            border-radius: 10px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .upload-area:hover {
            background: #eff6ff; /* Blue 50 */
        }
        .upload-area h3 {
            margin: 0 0 10px 0;
            font-size: 20px;
            font-weight: 500;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }
        .form-group select {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-color);
            font-size: 16px;
        }
        .form-group select:disabled {
            background-color: #e5e7eb;
            cursor: not-allowed;
        }
        .btn {
            background: var(--primary-color);
            color: white;
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 700;
            transition: background-color 0.2s;
        }
        .btn:hover {
            background: var(--primary-hover);
        }
        .processing-section, .results-section {
            display: none; /* Hidden by default */
        }
        .status-text {
            text-align: center;
            font-size: 18px;
            font-weight: 500;
            margin-bottom: 15px;
        }
        .progress-bar {
            width: 100%;
            height: 12px;
            background: var(--border-color);
            border-radius: 6px;
            overflow: hidden;
            margin: 20px 0 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: var(--primary-color);
            width: 0%;
            transition: width 0.5s;
        }
        .progress-text {
            text-align: center;
            color: var(--text-light);
        }
        .results-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .results-header h4 {
            font-size: 20px;
            margin: 0;
            font-weight: 400;
        }
        .results-header span {
            font-size: 28px;
            font-weight: 700;
            color: var(--primary-color);
        }
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .result-bar {
            background: var(--bg-color);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .result-bar .label {
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 10px;
        }
        .result-bar .value {
            font-size: 32px;
            font-weight: 700;
        }
        #healthyProbValue { color: #10b981; /* Green 500 */ }
        #sczProbValue { color: #ef4444; /* Red 500 */ }

    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Neuro<span>Scope</span></h1>
            <p>Advanced fMRI analysis for diagnostic insights.</p>
        </div>

        <!-- Main Card for Upload and Settings -->
        <div id="uploadCard" class="card">
            <div class="form-group">
                <label for="diseaseSelect">Analysis Target</label>
                <select id="diseaseSelect" disabled>
                    <option selected>Schizophrenia vs. Healthy</option>
                    <!-- <option>ADHD vs. Healthy (Coming Soon)</option> -->
                </select>
            </div>
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <h3>📁 Select fMRI File</h3>
                <p>Click here to choose a preprocessed .nii.gz file</p>
                <input type="file" id="fileInput" style="display: none;">
            </div>
        </div>

        <!-- Processing Card -->
        <div id="processingSection" class="processing-section card">
            <div class="status-text" id="statusText">Initializing...</div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="progress-text" id="progressText">0%</div>
        </div>

        <!-- Results Card -->
        <div id="resultsSection" class="results-section card">
            <div class="results-header">
                <h4>Primary Finding</h4>
                <span id="primaryDiagnosis"></span>
            </div>
            <div class="results-grid">
                <div class="result-bar">
                    <div class="label">Healthy</div>
                    <div class="value" id="healthyProbValue">0%</div>
                </div>
                <div class="result-bar">
                    <div class="label">Schizophrenia</div>
                    <div class="value" id="sczProbValue">0%</div>
                </div>
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
            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const data = await response.json();
                currentJobId = data.job_id;
                checkStatus();
            } catch (error) {
                alert('Upload failed: ' + error.message);
            }
        }

        async function checkStatus() {
            if (!currentJobId) return;
            try {
                const response = await fetch(`/status/${currentJobId}`);
                const data = await response.json();
                updateProgress(data.progress, data.status);
                if (data.status === 'completed') {
                    showResults(data.results);
                } else if (data.status === 'error') {
                    alert('Processing failed: ' + data.error);
                } else {
                    setTimeout(checkStatus, 1500);
                }
            } catch (error) {
                console.error('Status check failed:', error);
                setTimeout(checkStatus, 2000);
            }
        }

        function updateProgress(progress, status) {
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = Math.round(progress) + '%';
            const statusMap = {
                'preprocessing': 'Analyzing fMRI data structure...',
                'custom_processing': 'Applying advanced signal processing (NiLearn)...',
                'entropy': 'Extracting statistical features (Entropy)...',
                'prediction': 'Running diagnostic prediction model...',
                'completed': 'Analysis complete!'
            };
            document.getElementById('statusText').textContent = statusMap[status] || 'Processing...';
        }

        function showResults(results) {
            document.getElementById('processingSection').style.display = 'none';
            document.getElementById('resultsSection').style.display = 'block';

            document.getElementById('primaryDiagnosis').textContent = results.primary_diagnosis;

            // Note: The element IDs have changed to match the new design
            document.getElementById('healthyProbValue').textContent = results.probabilities.healthy.toFixed(1) + '%';
            document.getElementById('sczProbValue').textContent = results.probabilities.schizophrenia.toFixed(1) + '%';
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


# ==============================================================================
# === REAL ML PREDICTION FUNCTION ==============================================
# ==============================================================================
def run_ml_prediction(features):
    """
    Runs the real binary classification model.
    """
    if ML_MODEL is None:
        raise RuntimeError("ML Model is not loaded. Cannot perform prediction.")

    # Reshape the 1D feature array into a 2D array, as scikit-learn expects
    features_2d = features.reshape(1, -1)

    # Get the probabilities for [Class 0, Class 1] (e.g., [Healthy, SCZ])
    probabilities = ML_MODEL.predict_proba(features_2d)[0]

    # Get the primary diagnosis by finding the index of the highest probability
    max_idx = np.argmax(probabilities)
    primary_diagnosis = DIAGNOSES[max_idx]

    # Return the results in the format our new frontend expects
    return {
        'primary_diagnosis': primary_diagnosis,
        'probabilities': {
            'healthy': probabilities[0] * 100,
            'schizophrenia': probabilities[1] * 100
        }
    }


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        job_id = str(uuid.uuid4())
        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        jobs[job_id] = {'status': 'preprocessing', 'progress': 0, 'filepath': filepath}
        thread = threading.Thread(target=process_pipeline, args=(job_id,))
        thread.daemon = True
        thread.start()
        return jsonify({'job_id': job_id, 'message': 'Upload successful, processing started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# === MAIN PROCESSING PIPELINE (UPDATED) =======================================
# ==============================================================================
def process_pipeline(job_id):
    """
    Main processing pipeline - Uses real ML model at the end.
    """
    try:
        # We are still using the "Fast Check Mode" for development
        print("🚀 RUNNING IN FAST CHECK MODE - Skipping fMRIPrep! 🚀")
        preprocessed_data_dir = os.path.abspath('fast_check_data')
        jobs[job_id].update({'status': 'custom_processing', 'progress': 10})

        # Step 2: NiLearn processing
        final_processed_file = fmri_processing.run_nilearn_processing(preprocessed_data_dir, job_id)
        jobs[job_id].update({'status': 'entropy', 'progress': 40})

        # Step 3: Entropy calculation
        entropy_features = entropy_calculator.calculate_entropy_features(final_processed_file)
        jobs[job_id].update({'status': 'prediction', 'progress': 80})

        # Step 4: Run the REAL ML prediction
        results = run_ml_prediction(entropy_features)

        # Add a small delay for the final animation to look good
        time.sleep(2)

        # Complete
        jobs[job_id].update({'status': 'completed', 'progress': 100, 'results': results})

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
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])


if __name__ == '__main__':
    print("🧠 NeuroScope Server Starting...")
    print("📍 Open your browser to: http://localhost:5001")
    app.run(debug=True, host='localhost', port=5001)

