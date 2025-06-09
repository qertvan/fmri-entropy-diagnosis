# app.py - Simple Flask app for PyCharm
from flask import Flask, request, jsonify, render_template_string
import os
import time
import threading
import uuid
import random
import numpy as np
import fmri_processing
import entropy_calculator


app = Flask(__name__)

# Simple in-memory job storage
jobs = {}

# HTML template (embedded for simplicity)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>fMRI Diagnosis - Sample</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .upload-area { border: 2px dashed #007bff; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; }
        .upload-area:hover { background: #f8f9ff; }
        .btn { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #0056b3; }
        .progress-bar { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin: 20px 0; }
        .progress-fill { height: 100%; background: #007bff; width: 0%; transition: width 0.5s; }
        .results { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .prob-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }
        .prob-card { background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .prob-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .hidden { display: none; }
        .status { margin: 10px 0; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 NeuroScope</h1>
        <p>Upload an fMRI file to get started (simulated processing for demo)</p>

        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <h3>📁 Upload fMRI Data</h3>
            <p>Click here to select a file</p>
            <input type="file" id="fileInput" style="display: none;">
        </div>

        <div id="processingSection" class="hidden">
            <h3>🔄 Processing...</h3>
            <div class="status" id="statusText">Initializing...</div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div id="progressText">0%</div>
        </div>

        <div id="resultsSection" class="hidden">
            <h3>🎯 Diagnosis Results</h3>
            <div class="results">
                <h4>Primary Diagnosis: <span id="primaryDiagnosis"></span></h4>
                <p>Confidence: <span id="confidence"></span>%</p>

                <div class="prob-grid">
                    <div class="prob-card">
                        <div class="prob-value" id="healthyProb">0%</div>
                        <div>Healthy</div>
                    </div>
                    <div class="prob-card">
                        <div class="prob-value" id="schizophreniaProb">0%</div>
                        <div>Schizophrenia</div>
                    </div>
                    <div class="prob-card">
                        <div class="prob-value" id="adhdProb">0%</div>
                        <div>ADHD</div>
                    </div>
                    <div class="prob-card">
                        <div class="prob-value" id="bipolarProb">0%</div>
                        <div>Bipolar</div>
                    </div>
                </div>
            </div>

            <button class="btn" onclick="resetDemo()">Try Another File</button>
        </div>
    </div>

    <script>
        let currentJobId = null;

        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                uploadFile(file);
            }
        });

        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                currentJobId = data.job_id;

                document.getElementById('processingSection').classList.remove('hidden');
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
                    setTimeout(checkStatus, 1000);
                }

            } catch (error) {
                console.error('Status check failed:', error);
                setTimeout(checkStatus, 2000);
            }
        }

        function updateProgress(progress, status) {
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = progress + '%';
            document.getElementById('statusText').textContent = getStatusText(status);
        }

        function getStatusText(status) {
            const statusMap = {
                'preprocessing': 'Running fMRIPrep preprocessing...',
                'custom_processing': 'Custom preprocessing...',
                'entropy': 'Calculating entropy features...',
                'prediction': 'Running ML prediction...',
                'completed': 'Analysis complete!'
            };
            return statusMap[status] || 'Processing...';
        }

        function showResults(results) {
            document.getElementById('processingSection').classList.add('hidden');
            document.getElementById('resultsSection').classList.remove('hidden');

            document.getElementById('primaryDiagnosis').textContent = results.primary_diagnosis;
            document.getElementById('confidence').textContent = results.confidence.toFixed(1);

            document.getElementById('healthyProb').textContent = results.probabilities.healthy.toFixed(1) + '%';
            document.getElementById('schizophreniaProb').textContent = results.probabilities.schizophrenia.toFixed(1) + '%';
            document.getElementById('adhdProb').textContent = results.probabilities.adhd.toFixed(1) + '%';
            document.getElementById('bipolarProb').textContent = results.probabilities.bipolar.toFixed(1) + '%';
        }

        function resetDemo() {
            document.getElementById('processingSection').classList.add('hidden');
            document.getElementById('resultsSection').classList.add('hidden');
            document.getElementById('fileInput').value = '';
            currentJobId = null;
        }
    </script>
</body>
</html>
'''


# Simulated processing functions (replace with your actual code)
def simulate_fmriprep(filepath):
    """Simulate fMRIPrep processing"""
    time.sleep(2)  # Simulate processing time
    return f"preprocessed_{filepath}"


def simulate_preprocessing(filepath):
    """Simulate your custom preprocessing"""
    time.sleep(1.5)
    return np.random.randn(1000, 100)  # Simulate processed data


def simulate_entropy_calculation(data):
    """Simulate your entropy calculation"""
    time.sleep(1)
    return np.random.randn(50)  # Simulate entropy features


def simulate_ml_prediction(features):
    """
    Simulate ML prediction, but FIRST, let's inspect the features we received!
    """
    print("\n✅ --- ML Prediction Step Reached! --- ✅")
    print(f"Data type of features received: {type(features)}")
    print(f"Shape of the feature array: {features.shape}")

    # Let's check for any non-finite values (NaNs or Infs)
    if not np.all(np.isfinite(features)):
        print("🚨 WARNING: The feature array contains non-finite (NaN or Inf) values!")
    else:
        print("👍 The feature array contains all finite numbers.")

    # Print the first 10 entropy values to see what they look like
    print(f"First 10 entropy features: {features[:10]}")

    # The original simulation code can run after our checks
    print("--- Now running the original simulation... ---")
    time.sleep(1)
    probs = np.random.dirichlet([2, 1, 1, 1])
    diagnoses = ['Healthy', 'Schizophrenia', 'ADHD', 'Bipolar Disorder']
    max_idx = np.argmax(probs)
    return {
        'primary_diagnosis': diagnoses[max_idx],
        'confidence': probs[max_idx] * 100,
        'probabilities': {
            'healthy': probs[0] * 100,
            'schizophrenia': probs[1] * 100,
            'adhd': probs[2] * 100,
            'bipolar': probs[3] * 100
        }
    }

    # Generate realistic probabilities
    probs = np.random.dirichlet([2, 1, 1, 1])  # Favor healthy

    diagnoses = ['Healthy', 'Schizophrenia', 'ADHD', 'Bipolar Disorder']
    max_idx = np.argmax(probs)

    return {
        'primary_diagnosis': diagnoses[max_idx],
        'confidence': probs[max_idx] * 100,
        'probabilities': {
            'healthy': probs[0] * 100,
            'schizophrenia': probs[1] * 100,
            'adhd': probs[2] * 100,
            'bipolar': probs[3] * 100
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

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Save file (optional for demo)
        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Initialize job
        jobs[job_id] = {
            'status': 'preprocessing',
            'progress': 0,
            'filepath': filepath
        }

        # Start processing in background
        thread = threading.Thread(target=process_pipeline, args=(job_id,))
        thread.daemon = True
        thread.start()

        return jsonify({'job_id': job_id, 'message': 'Upload successful, processing started'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_pipeline(job_id):
    """
    Main processing pipeline - IN 'FAST CHECK' MODE
    This skips fMRIPrep and uses pre-processed data from a dedicated test folder.
    """
    try:
        # We still need the filepath variable, but we won't use it in this mode.
        filepath = jobs[job_id]['filepath']

        # === 'FAST CHECK' CONFIGURATION ===
        print("🚀 RUNNING IN FAST CHECK MODE - Skipping fMRIPrep! 🚀")
        # Set the path to the folder containing your preprocessed test data
        preprocessed_data_dir = os.path.abspath('fast_check_data')
        jobs[job_id].update({'status': 'preprocessing', 'progress': 35})  # Pretend fMRIPrep is done
        # =================================

        # Step 2: Custom NiLearn processing (This will run for real)
        # We pass the path to our 'fast_check_data' folder.
        jobs[job_id].update({'status': 'custom_processing', 'progress': 40})
        final_processed_file = fmri_processing.run_nilearn_processing(preprocessed_data_dir, job_id)
        jobs[job_id]['progress'] = 70

        # Step 3: Entropy calculation (This will run for real)
        jobs[job_id].update({'status': 'entropy', 'progress': 75})
        entropy_features = entropy_calculator.calculate_entropy_features(final_processed_file)
        jobs[job_id]['progress'] = 90

        # Step 4: Inspect the results
        results = simulate_ml_prediction(entropy_features)

        # Complete
        jobs[job_id].update({'status': 'completed', 'progress': 100, 'results': results})

    except Exception as e:
        # The enhanced error block to catch any failures
        import traceback
        print("\n" + "=" * 80)
        print("🚨🚨🚨 AN ERROR OCCURRED IN THE PIPELINE! 🚨🚨🚨")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80 + "\n")

        jobs[job_id].update({
            'status': 'error',
            'error': str(e)
        })


@app.route('/status/<job_id>')
def get_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(jobs[job_id])


if __name__ == '__main__':
    print("🧠 fMRI Diagnosis Sample Server Starting...")
    print("📍 Open your browser to: http://localhost:5001")
    print("🔄 Upload any file to see the simulated processing!")

    app.run(debug=True, host='localhost', port=5001)