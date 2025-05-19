import subprocess
from pathlib import Path
import shutil
import os
from config import MAIN_DIR

def download_file_with_wget(url, output_path=None, quiet=False, continue_download=True):

    wget_executable = shutil.which("wget")

    if not wget_executable:
        print("Error: wget is not installed or not found in PATH.")
        print("Please install wget (e.g., sudo apt-get install wget or brew install wget).")
        return False

    command = [wget_executable]

    if quiet:
        command.append("-q")

    if continue_download:
        command.append("-c")

    if output_path:

        output_dir = os.path.dirname(output_path)

        if output_dir and not os.path.exists(output_dir):

            try:
                os.makedirs(output_dir)
                print(f"Directory created: {output_dir}")
            except OSError as e:
                print(f"Error: Could not create output directory ({output_dir}): {e}")
                return False

        command.extend(["-O", output_path])

    command.append(url)

    try:

        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode == 0:

            if not quiet and result.stderr and " saved [" in result.stderr:
                print(result.stderr.strip().splitlines()[-1])
            return True

        else:
            print(f"Error: wget command failed. URL: {url}, Output Path: {output_path}")
            print(f"Exit code: {result.returncode}")
            print(f"wget error (stderr):\n{result.stderr.strip()}")

            if result.stdout:
                print(f"wget output (stdout):\n{result.stdout.strip()}")
            return False

    except FileNotFoundError:
        print(f"Error: '{wget_executable}' command not found.")
        return False

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False




OUT_PATH = Path(MAIN_DIR)
SAVE_PATH = OUT_PATH / "pretrained_model"


DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

DVAE_CHECKPOINT = os.path.join(SAVE_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(SAVE_PATH, os.path.basename(MEL_NORM_LINK))


files_to_download_dvae = []
if not os.path.isfile(DVAE_CHECKPOINT):
    files_to_download_dvae.append((DVAE_CHECKPOINT_LINK, DVAE_CHECKPOINT))
else:
    print(f"File already exists: {DVAE_CHECKPOINT}")

if not os.path.isfile(MEL_NORM_FILE):
    files_to_download_dvae.append((MEL_NORM_LINK, MEL_NORM_FILE))
else:
    print(f"File already exists: {MEL_NORM_FILE}")


if files_to_download_dvae:

    print("\n> Downloading DVAE files...")
    
    for url, out_file in files_to_download_dvae:

        print(f"  Downloading: {os.path.basename(out_file)}")

        if not download_file_with_wget(url, out_file, quiet=False, continue_download=True):
            print(f"  ERROR: Could not download {os.path.basename(out_file)}.")

    print("> DVAE file download/check completed.\n")



TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"


TOKENIZER_FILE = os.path.join(SAVE_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(SAVE_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))


files_to_download_xtts = []
if not os.path.isfile(TOKENIZER_FILE):
    files_to_download_xtts.append((TOKENIZER_FILE_LINK, TOKENIZER_FILE))
else:
    print(f"File already exists: {TOKENIZER_FILE}")

if not os.path.isfile(XTTS_CHECKPOINT):
    files_to_download_xtts.append((XTTS_CHECKPOINT_LINK, XTTS_CHECKPOINT))
else:
    print(f"File already exists: {XTTS_CHECKPOINT}")


if files_to_download_xtts:

    print("\n> Downloading XTTS v2.0 files...")

    for url, out_file in files_to_download_xtts:

        print(f"  Downloading: {os.path.basename(out_file)}")

        if not download_file_with_wget(url, out_file, quiet=False, continue_download=True):
            print(f"  ERROR: Could not download {os.path.basename(out_file)}.")

    print("> XTTS v2.0 file download/check completed.\n")


print("All required model files checked.")
print(f"DVAE checkpoint: {DVAE_CHECKPOINT}")
print(f"Mel normalization file: {MEL_NORM_FILE}")
print(f"Tokenizer file: {TOKENIZER_FILE}")
print(f"XTTS checkpoint: {XTTS_CHECKPOINT}")
