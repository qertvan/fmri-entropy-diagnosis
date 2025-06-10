# app.py - Updated for fMRI Diagnosis Workflow
from flask import Flask, request, jsonify, render_template_string
import os
import time
import threading
import uuid
import fmri_processing
import entropy_calculator
import run_ml_models  # YENİ

app = Flask(__name__)

# Basit, bellek-içi iş depolaması
jobs = {}

# --- YENİ HTML ARAYÜZÜ ---
# Kullanıcının hastalık ve veri türü seçmesine olanak tanır
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NeuroScope AI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; }
        .form-section { margin-bottom: 25px; }
        label { display: block; font-weight: bold; margin-bottom: 8px; }
        select, .file-input { width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ccc; }
        .radio-group label { display: inline-block; margin-right: 20px; font-weight: normal;}
        .upload-area { border: 2px dashed #007bff; padding: 30px; text-align: center; border-radius: 10px; margin-top: 15px; cursor: pointer; }
        .upload-area:hover { background: #f8f9ff; }
        .btn { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; width: 100%; }
        .btn:hover { background: #0056b3; }
        .progress-bar { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin: 20px 0; }
        .progress-fill { height: 100%; background: #28a745; width: 0%; transition: width 0.5s; }
        .results { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .hidden { display: none; }
        .status { margin: 10px 0; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 NeuroScope AI</h1>
        <p>Analyze fMRI data to predict neurological conditions.</p>

        <form id="uploadForm">
            <div class="form-section">
                <label for="diseaseSelect">1. Select Suspected Condition</label>
                <select id="diseaseSelect" name="selected_disease">
                    <option value="SCZ">Schizophrenia</option>
                    <option value="ADHD">ADHD</option>
                    <option value="BPD">Bipolar Disorder</option>
                </select>
            </div>

            <div class="form-section">
                <label>2. Select Data Type</label>
                <div class="radio-group">
                    <label><input type="radio" name="processing_mode" value="full" checked> Raw fMRI File (.nii or .nii.gz)</label>
                    <label><input type="radio" name="processing_mode" value="preprocessed"> Pre-processed Files</label>
                </div>
            </div>

            <div id="raw-upload" class="upload-area" onclick="document.getElementById('rawFileInput').click();">
                <h3>📁 Upload Raw fMRI Data</h3>
                <p>Click to select a single .nii or .nii.gz file</p>
                <input type="file" id="rawFileInput" name="raw_file" class="hidden">
                <span id="rawFileName">No file selected</span>
            </div>

            <div id="preprocessed-upload" class="hidden">
                <div class="upload-area" onclick="document.getElementById('boldFile').click();">
                    <h3>...preproc_bold.nii.gz</h3>
                    <input type="file" id="boldFile" name="bold_file" class="hidden">
                    <span id="boldFileName">No file selected</span>
                </div>
                <div class="upload-area" onclick="document.getElementById('confoundsFile').click();" style="margin-top:10px;">
                    <h3>...confounds.tsv</h3>
                    <input type="file" id="confoundsFile" name="confounds_file" class="hidden">
                    <span id="confoundsFileName">No file selected</span>
                </div>
            </div>

            <div style="margin-top:30px;">
                <button type="submit" class="btn">Start Analysis</button>
            </div>
        </form>

        <div id="processingSection" class="hidden">
            <h3>🔄 Processing... Please wait.</h3>
            <div class="status" id="statusText">Initializing...</div>
            <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
            <div id="progressText">0%</div>
        </div>

        <div id="resultsSection" class="hidden">
            <h3>🎯 Diagnosis Results</h3>
            <div class="results" id="resultsContent"></div>
            <button class="btn" onclick="resetDemo()">Analyze Another File</button>
        </div>
    </div>

<script>
    let currentJobId = null;

    // --- Form ve Dosya Yükleme Arayüzü Mantığı ---
    const form = document.getElementById('uploadForm');
    const rawUploadDiv = document.getElementById('raw-upload');
    const preprocessedUploadDiv = document.getElementById('preprocessed-upload');

    document.querySelectorAll('input[name="processing_mode"]').forEach(radio => {
        radio.addEventListener('change', function() {
            rawUploadDiv.classList.toggle('hidden', this.value !== 'full');
            preprocessedUploadDiv.classList.toggle('hidden', this.value !== 'preprocessed');
        });
    });

    // Dosya isimlerini gösterme
    document.getElementById('rawFileInput').addEventListener('change', e => { document.getElementById('rawFileName').textContent = e.target.files[0]?.name || 'No file selected'; });
    document.getElementById('boldFile').addEventListener('change', e => { document.getElementById('boldFileName').textContent = e.target.files[0]?.name || 'No file selected'; });
    document.getElementById('confoundsFile').addEventListener('change', e => { document.getElementById('confoundsFileName').textContent = e.target.files[0]?.name || 'No file selected'; });

    // Form gönderimini ele alma
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = new FormData(form);
        const processingMode = formData.get('processing_mode');

        // Dosya kontrolü
        if (processingMode === 'full' && !formData.get('raw_file')?.name) {
            alert('Please select a raw fMRI file.'); return;
        }
        if (processingMode === 'preprocessed' && (!formData.get('bold_file')?.name || !formData.get('confounds_file')?.name)) {
            alert('Please select both the bold and confounds files.'); return;
        }

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);

            const data = await response.json();
            currentJobId = data.job_id;

            form.classList.add('hidden');
            document.getElementById('processingSection').classList.remove('hidden');
            checkStatus();

        } catch (error) {
            alert('Upload failed: ' + error.message);
        }
    });

    async function checkStatus() {
        if (!currentJobId) return;
        try {
            const response = await fetch(`/status/${currentJobId}`);
            const data = await response.json();

            updateProgress(data.progress, data.status_text);

            if (data.status === 'completed') {
                showResults(data.results);
            } else if (data.status === 'error') {
                alert('Processing failed: ' + data.error);
                resetDemo();
            } else {
                setTimeout(checkStatus, 2000); // 2 saniyede bir kontrol et
            }
        } catch (error) {
            console.error('Status check failed:', error);
            setTimeout(checkStatus, 5000); // Hata durumunda daha uzun bekle
        }
    }

    function updateProgress(progress, status) {
        document.getElementById('progressFill').style.width = progress + '%';
        document.getElementById('progressText').textContent = Math.round(progress) + '%';
        document.getElementById('statusText').textContent = status;
    }

    function showResults(results) {
        document.getElementById('processingSection').classList.add('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');
        const disease = Object.keys(results.probabilities).find(k => k !== 'healthy');

        const content = `
            <h4>Primary Diagnosis: <span style="color: #007bff;">${results.primary_diagnosis}</span></h4>
            <p>Confidence: <strong>${results.confidence.toFixed(1)}%</strong></p>
            <hr>
            <h4>Probability Distribution</h4>
            <p>Healthy: ${results.probabilities.healthy.toFixed(1)}%</p>
            <p>${disease}: ${results.probabilities[disease].toFixed(1)}%</p>
        `;
        document.getElementById('resultsContent').innerHTML = content;
    }

    function resetDemo() {
        document.getElementById('processingSection').classList.add('hidden');
        document.getElementById('resultsSection').classList.add('hidden');
        form.reset();
        form.classList.remove('hidden');
        rawUploadDiv.classList.remove('hidden');
        preprocessedUploadDiv.classList.add('hidden');
        document.getElementById('rawFileName').textContent = 'No file selected';
        document.getElementById('boldFileName').textContent = 'No file selected';
        document.getElementById('confoundsFileName').textContent = 'No file selected';
        currentJobId = null;
    }
</script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/upload', methods=['POST'])
def upload_file_handler():
    try:
        processing_mode = request.form.get('processing_mode')
        selected_disease = request.form.get('selected_disease')

        if not processing_mode or not selected_disease:
            return jsonify({'error': 'Missing form data'}), 400

        job_id = str(uuid.uuid4())
        job_dir = os.path.join('uploads', job_id)
        os.makedirs(job_dir, exist_ok=True)

        job_info = {
            'status': 'queued', 'progress': 0, 'status_text': 'Job is queued...',
            'processing_mode': processing_mode, 'selected_disease': selected_disease, 'filepaths': {}
        }

        if processing_mode == 'full':
            file = request.files.get('raw_file')
            if not file or file.filename == '': return jsonify({'error': 'No raw file uploaded'}), 400

            raw_filepath = os.path.join(job_dir, file.filename)
            file.save(raw_filepath)
            job_info['filepaths']['raw'] = raw_filepath

        elif processing_mode == 'preprocessed':
            bold_file = request.files.get('bold_file')
            confounds_file = request.files.get('confounds_file')

            if not bold_file or not confounds_file: return jsonify(
                {'error': 'Both preprocessed files are required'}), 400

            preproc_input_dir = os.path.join(job_dir, 'preproc_input')
            os.makedirs(preproc_input_dir, exist_ok=True)

            bold_filepath = os.path.join(preproc_input_dir, bold_file.filename)
            confounds_filepath = os.path.join(preproc_input_dir, confounds_file.filename)
            bold_file.save(bold_filepath)
            confounds_file.save(confounds_filepath)
            job_info['filepaths']['preproc_dir'] = preproc_input_dir

        jobs[job_id] = job_info
        thread = threading.Thread(target=process_pipeline, args=(job_id,))
        thread.daemon = True
        thread.start()

        return jsonify({'job_id': job_id, 'message': 'Processing started'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def process_pipeline(job_id):
    job_info = jobs[job_id]
    processing_mode = job_info['processing_mode']
    selected_disease = job_info['selected_disease']

    def update_status(status, progress, text):
        jobs[job_id].update({'status': status, 'progress': progress, 'status_text': text})

    try:
        nilearn_input_dir = None

        if processing_mode == 'full':
            # --- Adım 1: fMRIPrep (Eğer ham veri yüklendiyse) ---
            update_status('fmriprep', 5, 'Starting fMRIPrep... this may take a very long time.')
            raw_filepath = job_info['filepaths']['raw']
            # NOT: run_fmriprep'in çıktısı 'preproc_clean' klasörünün yolu olmalıdır.
            nilearn_input_dir = fmri_processing.run_fmriprep(raw_filepath, job_id)
            update_status('fmriprep', 40, 'fMRIPrep completed.')

        elif processing_mode == 'preprocessed':
            # --- Adım 1: fMRIPrep atlanır ---
            update_status('nilearn', 40, 'fMRIPrep skipped. Starting post-processing.')
            nilearn_input_dir = job_info['filepaths']['preproc_dir']

        # --- Adım 2: NiLearn ile Son İşleme ---
        update_status('nilearn', 50, 'Running NiLearn post-processing...')
        final_processed_file = fmri_processing.run_nilearn_processing(nilearn_input_dir, job_id)
        update_status('nilearn', 70, 'NiLearn processing complete.')

        # --- Adım 3: Entropi Özelliklerini Hesaplama ---
        update_status('entropy', 75, 'Calculating entropy features...')
        entropy_features = entropy_calculator.calculate_entropy_features(final_processed_file)
        update_status('entropy', 90, 'Entropy calculation complete.')

        # --- Adım 4: Makine Öğrenmesi ile Tahmin ---
        update_status('prediction', 95, f'Running prediction model for {selected_disease}...')
        results = run_ml_models.run_prediction(entropy_features, selected_disease)

        # --- Bitiş ---
        update_status('completed', 100, 'Analysis complete!')
        jobs[job_id]['results'] = results

    except Exception as e:
        import traceback
        print(f"\n--- PIPELINE ERROR for job {job_id} ---")
        traceback.print_exc()
        print("--- END ERROR ---")
        jobs[job_id].update({'status': 'error', 'error': str(e)})


@app.route('/status/<job_id>')
def get_status(job_id):
    return jsonify(jobs.get(job_id, {'error': 'Job not found', 'status': 'error'}))


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5001)