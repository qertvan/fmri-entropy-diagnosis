# NeuroScope: fMRI Analysis Web Application

NeuroScope is a proof-of-concept web application built with Flask that provides a user-friendly interface for advanced fMRI analysis. It allows users to upload fMRI data and receive diagnostic insights based on pre-trained machine learning models. The application is designed to be scalable, supporting multiple simultaneous analyses through asynchronous background processing.

## Features

-   **Intuitive Web Interface:** A clean, modern UI for easy file upload and progress monitoring.
-   **Multiple Data Formats:** Supports both pre-processed fMRI files (`.nii.gz` + `.tsv`) and raw BIDS datasets.
-   **Asynchronous Processing:** Long-running analyses are handled in background threads, so the UI remains responsive.
-   **Real-time Status Polling:** The frontend periodically checks the job status to provide real-time feedback and a progress bar.
-   **Multi-Disease Support:** The architecture supports multiple machine learning models for different diagnostic targets (e.g., Schizophrenia, ADHD, Bipolar Disorder).
-   **Automatic Cleanup:** Temporarily uploaded and processed files are automatically deleted after analysis is complete.

## Project Structure

```
.
├── app.py                    # The main Flask application file
├── fmri_processing.py        # Module for fMRIPrep and NiLearn post-processing
├── entropy_calculator.py     # Module for calculating entropy features
├── ml_predictor.py           # Module for loading models and running predictions
├── models/                   # Directory to store pre-trained model files (.joblib)
│   ├── final_model_ADHD.joblib
│   ├── final_model_BPD.joblib
│   └── final_model_SCZ.joblib
├── static/                   # Directory for logo
│   └── logo.png
├── uploads/                  # Temporary directory for user-uploaded files
└── requirements.txt          # Python package dependencies
```

## Setup and Installation

**Prerequisites:**
-   Python 3.8+
-   `fmriprep` and its dependencies installed in your environment if you intend to process raw data.

**Instructions:**

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place model files:** Ensure your pre-trained machine learning models (`.pkl` files) are placed inside the `models/` directory. The filenames should match those expected by `ml_predictor.py`.

## How to Run

1.  **Start the Flask server:**
    ```bash
    python app.py
    ```

2.  **Open the application:** Once the server is running, you will see a message in the terminal. Open your web browser and navigate to:
    [http://localhost:5001](http://localhost:5001)

## API Endpoints

The application exposes the following API endpoints:

#### `GET /`

-   **Description:** Serves the main HTML page of the application.
-   **Response:** `text/html`

#### `POST /upload`

-   **Description:** The main endpoint for uploading files and starting an analysis job.
-   **Request Body:** `multipart/form-data`
    -   `data_type` (str): The type of data being uploaded. Either `"preprocessed"` or `"raw"`.
    -   `disease` (str): The key for the target disease model (e.g., `"scz"`, `"adhd"`).
    -   `fmri_file` (file): The `.nii` or `.nii.gz` file (required if `data_type` is `"preprocessed"`).
    -   `tsv_file` (file): The `.tsv` confounds file (required if `data_type` is `"preprocessed"`).
    -   `bids_files` (file list): The list of all files in the BIDS dataset (required if `data_type` is `"raw"`).
-   **Success Response (200 OK):**
    ```json
    {
      "job_id": "a1b2c3d4-e5f6-a7b8-c9d0-e1f2a3b4c5d6",
      "message": "Upload successful, processing started"
    }
    ```
-   **Error Response (4xx/5xx):**
    ```json
    {
      "error": "A descriptive error message."
    }
    ```

#### `GET /status/<job_id>`

-   **Description:** Polls for the status of a specific job.
-   **URL Parameter:**
    -   `job_id` (str): The unique ID returned by the `/upload` endpoint.
-   **Success Response (200 OK):**
    ```json
    {
      "status": "entropy",
      "progress": 70,
      "data_type": "raw",
      "bids_dir": "uploads/a1b2c3d4-e5f6-a7b8-c9d0-e1f2a3b4c5d6"
    }
    ```
    Or, upon completion:
    ```json
    {
        "status": "completed",
        "progress": 100,
        "results": {
            "primary_diagnosis": "Schizophrenia",
            "class_names": ["Healthy", "Schizophrenia"],
            "probabilities": {
                "healthy": 15.8,
                "schizophrenia": 84.2
            }
        }
    }
    ```
-   **Error Response (404 Not Found):**
    ```json
    {
      "error": "Job not found"
    }
    ```
