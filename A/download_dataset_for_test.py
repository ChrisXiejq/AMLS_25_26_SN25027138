import os
import urllib.request

def get_dataset_root():
    """
    Get the root directory for the BreastMNIST dataset.
    Returns:
        str: Path to the dataset root directory.
    """
    dataset_root = os.path.join("./", "Datasets", "BreastMNIST")
    os.makedirs(dataset_root, exist_ok=True)
    return dataset_root


def download_file(url, save_path):
    """
    Download a file from a URL to a specified local path.
    Args:
        url (str): URL of the file to download.
        save_path (str): Local path to save the downloaded file.
    """
    print(f"Downloading:\n  {url}\n→ {save_path}")
    urllib.request.urlretrieve(url, save_path)
    print("Done.")


def prepare_breastmnist_dataset():
    """
    Download ONLY breastmnist.npz for local testing.
    Returns:
        str: Path to the dataset root directory.
    """
    root = get_dataset_root()

    # Needed file
    npz_path = os.path.join(root, "breastmnist.npz")

    # Official dataset link
    npz_url = "https://zenodo.org/records/10519652/files/breastmnist.npz"

    if os.path.exists(npz_path):
        print(f"BreastMNIST dataset already exists at: {root}")
        return root

    print("BreastMNIST dataset not found. Downloading...")

    # download ONLY the .npz file
    download_file(npz_url, npz_path)

    print("BreastMNIST dataset downloaded successfully.")
    return root
