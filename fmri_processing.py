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
# === PART 1: FMRIPREP FUNCTION (MODIFIED) =====================================
# ==============================================================================

BIDS_FILENAME_TEMPLATE = 'sub-{subject_id}_ses-01_task-rest_bold.nii.gz'


def run_fmriprep(uploaded_filepath, job_id):
    """
    Runs fMRIPrep and then calls the cleaning function to prepare for NiLearn.
    """
    # ... (The first part of the function preparing directories and the command is unchanged) ...
    print("--- Starting fMRIPrep Step ---")
    bids_input_dir = os.path.abspath(f'bids_input/{job_id}')
    func_dir = os.path.join(bids_input_dir, 'sub-01', 'func')
    os.makedirs(func_dir, exist_ok=True)
    bids_filepath = os.path.join(func_dir, BIDS_FILENAME_TEMPLATE.format(subject_id='01'))
    shutil.copy(uploaded_filepath, bids_filepath)
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
        '--participant-label', '01', '--fs-license-file', '/opt/freesurfer/license.txt',
        '--output-spaces', 'MNI152NLin2009cAsym', '--skip-bids-validation',
        '--nthreads', '2', '--mem_mb', '10000'
    ]
    print("Executing fMRIPrep command...")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("fMRIPrep completed successfully.")

        # --- THIS IS THE NEW PART ---
        # Define source and target directories for the cleaning step
        raw_fmriprep_dir = os.path.join(output_dir, 'fmriprep')
        clean_preproc_dir = os.path.join(output_dir, 'preproc_clean')

        # Run the cleaning function
        clean_and_organize_fmriprep_output(
            source_fmriprep_dir=raw_fmriprep_dir,
            target_clean_dir=clean_preproc_dir,
            subject_id='01'
        )

        # **IMPORTANT**: Return the path to the NEW, CLEANED directory
        return clean_preproc_dir

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


# --- MAIN NILEARN FUNCTION (MODIFIED) ---
# In fmri_processing.py

# --- MAIN NILEARN FUNCTION (UPDATED FOR FLEXIBILITY) ---
def run_nilearn_processing(input_data_dir, job_id, subject_id='01', tr=2.0):
    """
    This function now reads from a directory containing the bold and confounds files.
    """
    print("\n--- Starting NiLearn Post-Processing Step ---")

    # --- THIS LOGIC IS NOW MORE FLEXIBLE ---
    # Find the bold and confounds files directly within the given input directory
    # It will work for both the 'preproc_clean' folder and our new 'fast_check_data' folder.
    bold_files = glob.glob(os.path.join(input_data_dir, f'*-preproc_bold.nii.gz'))
    confounds_files = glob.glob(os.path.join(input_data_dir, f'*-confounds_timeseries.tsv'))

    if not bold_files: raise FileNotFoundError(f"BOLD file not found in input directory: {input_data_dir}")
    if not confounds_files: raise FileNotFoundError(f"Confounds file not found in input directory: {input_data_dir}")

    bold_path = bold_files[0]
    confounds_path = confounds_files[0]
    # ------------------------------------

    # Define a new directory for NiLearn outputs within the main job output folder
    # We find the main 'outputs/{job_id}' folder to keep things organized
    main_output_dir = os.path.abspath(f'outputs/{job_id}')
    nilearn_output_dir = os.path.join(main_output_dir, 'nilearn_output')
    os.makedirs(nilearn_output_dir, exist_ok=True)

    print(f"\nProcessing files from: {input_data_dir}")
    print(f"Input BOLD: {bold_path}")
    print(f"Output Dir: {nilearn_output_dir}")

    # The rest of the NiLearn pipeline is unchanged
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