import os
import shutil
import subprocess
import glob  # Added for the cleaning step
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.interpolate import interp1d
from nilearn.image import clean_img, smooth_img, resample_to_img, new_img_like
from nilearn.signal import clean
from nilearn.datasets import load_mni152_template


# =================================================================================
# === NEW HELPER FUNCTION FOR CLEANING (Adapted from your script) =================
# =================================================================================
def clean_and_organize_fmriprep_output(source_fmriprep_dir, target_clean_dir, subject_id):
    """
    Finds the essential files from the raw fMRIPrep output and copies them
    to a new, clean directory for the next pipeline step.
    """
    print("\n--- Starting fMRIPrep Output Cleaning Step ---")
    os.makedirs(target_clean_dir, exist_ok=True)

    # Define the files we want to find and keep
    file_patterns = [
        f"sub-{subject_id}*_desc-confounds_timeseries.tsv",
        f"sub-{subject_id}*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    ]

    found_files = []
    for pattern in file_patterns:
        # Search recursively inside the source fMRIPrep directory
        search_path = os.path.join(source_fmriprep_dir, '**', pattern)
        matching_files = glob.glob(search_path, recursive=True)

        if not matching_files:
            raise FileNotFoundError(
                f"Cleaning step failed: Could not find any files matching pattern '{pattern}' inside {source_fmriprep_dir}")

        # Copy the found file to the clean directory
        for file_path in matching_files:
            filename = os.path.basename(file_path)
            destination = os.path.join(target_clean_dir, filename)

            # Using copy is safer than move, as it leaves the original fMRIPrep output intact for debugging
            shutil.copy(file_path, destination)
            print(f"Copied essential file to: {destination}")
            found_files.append(destination)

    print("✅ Cleaning step complete.")
    return found_files


# ==============================================================================
# === PART 1: FMRIPREP FUNCTION (MODIFIED FOR BIDS DIR) ========================
# ==============================================================================

def run_fmriprep(bids_dir, job_id):
    """
    Runs fMRIPrep on a BIDS directory and returns the cleaned preprocessed bold and confounds file paths.
    """
    print("--- Starting fMRIPrep Step ---")
    bids_input_dir = os.path.abspath(bids_dir)
    output_dir = os.path.abspath(f'outputs/{job_id}')
    os.makedirs(output_dir, exist_ok=True)
    FREESURFER_LICENSE_PATH = os.path.abspath('license.txt')
    if not os.path.exists(FREESURFER_LICENSE_PATH):
        raise FileNotFoundError("License file not found!")
    command = [
        'docker', 'run', '--rm', '--platform', 'linux/amd64', '-e', 'KMP_AFFINITY=disabled',
        '-v', f'{bids_input_dir}:/data:ro', '-v', f'{output_dir}:/out',
        '-v', f'{FREESURFER_LICENSE_PATH}:/opt/freesurfer/license.txt',
        'nipreps/fmriprep:25.0.0', '/data', '/out', 'participant',
        '--fs-license-file', '/opt/freesurfer/license.txt',
        '--output-spaces', 'MNI152NLin2009cAsym', '--skip-bids-validation',
        '--nthreads', '2', '--mem_mb', '10000'
    ]
    print("Executing fMRIPrep command...")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("fMRIPrep completed successfully.")
        # --- Clean output ---
        raw_fmriprep_dir = os.path.join(output_dir, 'fmriprep')
        clean_preproc_dir = os.path.join(output_dir, 'preproc_clean')
        found_files = clean_and_organize_fmriprep_output(
            source_fmriprep_dir=raw_fmriprep_dir,
            target_clean_dir=clean_preproc_dir,
            subject_id='01'
        )
        # Find the cleaned files
        bold_files = glob.glob(os.path.join(clean_preproc_dir, '*preproc_bold.nii.gz'))
        confounds_files = glob.glob(os.path.join(clean_preproc_dir, '*confounds_timeseries.tsv'))
        if not bold_files or not confounds_files:
            raise FileNotFoundError("Could not find cleaned preprocessed or confounds files after fmriprep.")
        return bold_files[0], confounds_files[0]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("--- A critical error occurred during fMRIPrep or Cleaning! ---")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Return code: {e.returncode}")
            print("STDERR:", e.stderr)
        else:
            print("Error:", e)
        raise e


# ==============================================================================
# === PART 2: NILEARN PROCESSING FUNCTIONS (MODIFIED) ==========================
# ==============================================================================

# --- (Helper functions like scrub_fd, interpolate_scrubbed, etc., are unchanged) ---
def load_data(bold_path, confounds_path):
    img = nib.load(bold_path)
    confounds_df = pd.read_csv(confounds_path, sep='\t')
    return img, confounds_df


# ... (all other helper functions remain the same) ...
def scrub_fd(data, confounds_df, threshold=0.2):
    fd = confounds_df['FramewiseDisplacement'].fillna(0).values
    bad_trs = np.where(fd > threshold)[0]
    scrub_idx = set()
    for idx in bad_trs:
        scrub_idx.update([idx - 1, idx, idx + 1, idx + 2])
    return sorted([i for i in scrub_idx if 0 <= i < data.shape[3]])


def interpolate_scrubbed(data, scrub_idx, affine, header):
    n_voxels = np.prod(data.shape[:3])
    flat_data = data.reshape((n_voxels, data.shape[3]))
    good_indices = np.setdiff1d(np.arange(data.shape[3]), scrub_idx)
    if len(good_indices) < 2:
        return nib.Nifti1Image(data, affine=affine, header=header)
    for v in range(flat_data.shape[0]):
        ts = flat_data[v]
        interp = interp1d(good_indices, ts[good_indices], kind='linear', fill_value='extrapolate')
        flat_data[v, scrub_idx] = interp(scrub_idx)
    interpolated = flat_data.reshape(data.shape)
    return nib.Nifti1Image(interpolated, affine=affine, header=header)


def get_nuisance_regressors(confounds_df):
    base_cols = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'WhiteMatter', 'GlobalSignal']
    compcor_cols = [col for col in confounds_df.columns if col.startswith("a_comp_cor_")][:5]
    selected_cols = base_cols + compcor_cols
    existing_cols = [col for col in selected_cols if col in confounds_df.columns]
    return confounds_df[existing_cols].fillna(0)


def regress_out(img, confounds_df, tr):
    nuisance_regressors = get_nuisance_regressors(confounds_df)
    return clean_img(img, confounds=nuisance_regressors.values, detrend=True, standardize=False, t_r=tr)


def smooth_image(img, fwhm=6):
    return smooth_img(img, fwhm=fwhm)


def bandpass_filter(img, tr, low_pass=0.08, high_pass=0.009):
    data_filtered = clean(img.get_fdata(), t_r=tr, low_pass=low_pass, high_pass=high_pass, detrend=False,
                          standardize=False)
    return new_img_like(img, data_filtered)


def cleanup_temp_files(output_dir, filenames):
    for fname in filenames:
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            os.remove(path)
            print(f"Cleaned up temporary file: {fname}")


# --- MAIN NILEARN FUNCTION (UPDATED FOR FLEXIBILITY) ---
def run_nilearn_processing(input_data, job_id, tsv_path=None, tr=2.0):
    """
    If input_data is a directory, auto-find bold/confounds files. If it's a file, use it directly (tsv_path required).
    Returns the final processed bold file path.
    """
    print("\n--- Starting NiLearn Post-Processing Step ---")
    if os.path.isdir(input_data):
        bold_files = glob.glob(os.path.join(input_data, '*preproc_bold.nii.gz'))
        confounds_files = glob.glob(os.path.join(input_data, '*confounds_timeseries.tsv'))
        if not bold_files:
            raise FileNotFoundError(f"BOLD file not found in input directory: {input_data}")
        if not confounds_files:
            raise FileNotFoundError(f"Confounds file not found in input directory: {input_data}")
        bold_path = bold_files[0]
        confounds_path = confounds_files[0]
    else:
        if tsv_path is None:
            raise ValueError("tsv_path is required when input_data is a file path.")
        bold_path = input_data
        confounds_path = tsv_path
    # Output dir
    main_output_dir = os.path.abspath(f'outputs/{job_id}')
    nilearn_output_dir = os.path.join(main_output_dir, 'nilearn_output')
    os.makedirs(nilearn_output_dir, exist_ok=True)
    print(f"\nProcessing files from: {input_data}")
    print(f"Input BOLD: {bold_path}")
    print(f"Output Dir: {nilearn_output_dir}")
    # Pipeline
    img, confounds_df = load_data(bold_path, confounds_path)
    data = img.get_fdata()
    scrub_idx = scrub_fd(data, confounds_df)
    interpolated_img = interpolate_scrubbed(data, scrub_idx, img.affine, img.header)
    regressed_img = regress_out(interpolated_img, confounds_df, tr)
    template_3mm = load_mni152_template(resolution=3)
    resampled_img = resample_to_img(regressed_img, template_3mm, interpolation='continuous')
    smoothed_img = smooth_image(resampled_img, fwhm=6.0)
    final_img = bandpass_filter(smoothed_img, tr, low_pass=0.08, high_pass=0.009)
    final_path = os.path.join(nilearn_output_dir, "bold_final_processed.nii.gz")
    final_img.to_filename(final_path)
    print(f"✅ NiLearn processing complete. Final file at: {final_path}")
    return final_path