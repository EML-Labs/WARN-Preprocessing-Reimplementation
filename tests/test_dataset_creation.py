import h5py
import os
import numpy as np
import pytest

MATLAB_DATASET_DIR = "rp_data/"
PYTHON_DATASET_DIR = "rp_data_python/"

PER_SLICE_MSE_THRESHOLD = 1e-6  # start strict, relax later if needed


@pytest.mark.parametrize("sample_name", [
    "1_physionet",
    "2_physionet",
    "3_physionet",
    "4_physionet",
    "5_physionet",
])
def test_dataset_existence_and_shape(sample_name):
    matlab_file = os.path.join(MATLAB_DATASET_DIR, sample_name + ".hdf5")
    python_file = os.path.join(PYTHON_DATASET_DIR, sample_name + ".hdf5")

    assert os.path.exists(matlab_file)
    assert os.path.exists(python_file)

    with h5py.File(matlab_file, "r") as f_m, h5py.File(python_file, "r") as f_p:
        x_m = f_m["/x"][:]
        y_m = f_m["/y"][:]
        x_p = f_p["/x"][:]
        y_p = f_p["/y"][:]

    assert x_m.shape == x_p.shape == (138, 224, 224)
    assert y_m.shape == y_p.shape
    assert x_m.dtype == x_p.dtype
    assert y_m.dtype == y_p.dtype


@pytest.mark.parametrize("sample_name", [
    "1_physionet",
    "2_physionet",
    "3_physionet",
    "4_physionet",
    "5_physionet",
])
def test_per_slice_mse(sample_name):
    matlab_file = os.path.join(MATLAB_DATASET_DIR, sample_name + ".hdf5")
    python_file = os.path.join(PYTHON_DATASET_DIR, sample_name + ".hdf5")

    with h5py.File(matlab_file, "r") as f_m, h5py.File(python_file, "r") as f_p:
        x_m = f_m["/x"][:].astype(np.float64)
        x_p = f_p["/x"][:].astype(np.float64)

    # Per-slice MSE (138 slices)
    per_slice_mse = np.mean((x_m - x_p) ** 2, axis=(1, 2))

    print(
        f"[{sample_name}] "
        f"mean MSE={per_slice_mse.mean():.3e}, "
        f"max MSE={per_slice_mse.max():.3e}"
    )

    bad = np.where(per_slice_mse > PER_SLICE_MSE_THRESHOLD)[0]

    assert len(bad) == 0, (
        f"{sample_name}: {len(bad)} slices exceed threshold "
        f"{PER_SLICE_MSE_THRESHOLD:.1e}. "
        f"Worst slice={bad[0]}, "
        f"MSE={per_slice_mse[bad[0]]:.3e}"
    )
