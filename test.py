import os
import h5py
import numpy as np
from skimage.metrics import structural_similarity as ssim

MATLAB_DATASET_DIR = "rp_data/"
PYTHON_DATASET_DIR = "rp_data_python/"

matlab_files = sorted([f for f in os.listdir(MATLAB_DATASET_DIR) if f.endswith(".hdf5")])
python_files = sorted([f for f in os.listdir(PYTHON_DATASET_DIR) if f.endswith(".hdf5")])

print(f"Found {len(matlab_files)} MATLAB files and {len(python_files)} Python files.")

# for matlab_file, python_file in zip(matlab_files, python_files):
#     matlab_path = os.path.join(MATLAB_DATASET_DIR, matlab_file)
#     python_path = os.path.join(PYTHON_DATASET_DIR, python_file)

#     print(f"Comparing {matlab_file} and {python_file}...")

#     with h5py.File(matlab_path, "r") as f_m, h5py.File(python_path, "r") as f_p:
#         x_m = f_m["/x"][:]
#         y_m = f_m["/y"][:]
#         x_p = f_p["/x"][:]
#         y_p = f_p["/y"][:]


#     # Per-slice MSE (138 slices)
#     per_slice_mse = np.mean((x_m.astype(np.float64) - x_p.astype(np.float64)) ** 2, axis=(1, 2))

#     print(
#         f"[{matlab_file}] "
#         f"mean MSE={per_slice_mse.mean():.3e}, "
#         f"max MSE={per_slice_mse.max():.3e}"
#     )

def normalize_pixels(img):
    img = img.astype('float32')
    # Subtract mean (centers the histogram at 0)
    img -= np.mean(img)
    # Divide by std dev (scales contrast to 1.0)
    # Adding 1e-6 to avoid divide by zero for blank images
    img /= (np.std(img) + 1e-6)
    return img


def test_image_structure(img_matlab, img_python):
    # Ensure images are at least 2D arrays
    # img_matlab and img_python should be your loaded arrays (0-255)
    
    score, diff = ssim(img_matlab, img_python, full=True, data_range=255)
    
    print(f"SSIM Score: {score}")
    
    # SSIM of 1.0 means identical. 
    # > 0.90 is usually considered "structurally equivalent" for different renderers.
    # assert score > 0.95, f"Structure mismatch! SSIM only {score}"

random_index = np.random.randint(0, len(python_files))
print(f"Random slice index for visual inspection: {random_index}")

matlab_file = os.path.join(MATLAB_DATASET_DIR, matlab_files[random_index])
python_file = os.path.join(PYTHON_DATASET_DIR, python_files[random_index])

with h5py.File(matlab_file, "r") as f_m, h5py.File(python_file, "r") as f_p:
    x_m = f_m["/x"][:]
    y_m = f_m["/y"][:]
    x_p = f_p["/x"][:]
    y_p = f_p["/y"][:]

    random_slice = np.random.randint(0, x_m.shape[0])
    print(f"Inspecting slice {random_slice} from file {matlab_files[random_index]}")

    print("MATLAB slice data (min, max, mean):", x_m[random_slice].min(), x_m[random_slice].max(), x_m[random_slice].mean())
    print("Python slice data (min, max, mean):", x_p[random_slice].min(), x_p[random_slice].max(), x_p[random_slice].mean())

    norm_mat = normalize_pixels(x_m[random_slice])
    norm_py = normalize_pixels(x_p[random_slice])

    mse = np.mean((norm_mat - norm_py) ** 2)
    print(f"Normalized MSE for slice {random_slice}: {mse:.3e}")

    test_image_structure(x_m[random_slice], x_p[random_slice])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"MATLAB Slice {random_slice}")
    plt.imshow(x_m[random_slice], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(f"Python Slice {random_slice}")
    plt.imshow(x_p[random_slice], cmap='gray')
    plt.show()