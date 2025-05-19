import torch
import sys
import torchaudio
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


SCRIPT_DIR = Path("/content/XTTS_V2")

PRETRAINED_MODEL_ROOT = SCRIPT_DIR / "pretrained_model"
OUTPUT_ROOT_DIR = SCRIPT_DIR / "training_output"
SPEAKER_REFERENCE_ROOT = SCRIPT_DIR / "speaker_reference"

CHECKPOINT_MODEL_PATH = OUTPUT_ROOT_DIR / "checkpoints" / "xtts_finetune_v2-May-19-2025_04+06PM-af643bc"

CONFIG_CHECKPOINT_PATH = CHECKPOINT_MODEL_PATH / "config.json"
XTTS_CHECKPOINT_PATH = CHECKPOINT_MODEL_PATH / "best_model_338.pth"

TOKENIZER_PATH = PRETRAINED_MODEL_ROOT / "vocab.json"

SPEAKER_REFERENCE_WAV_PATH = SPEAKER_REFERENCE_ROOT / "reference.wav"
OUTPUT_WAV_PATH = OUTPUT_ROOT_DIR / "samples" / "output.wav"


def check_input_files_exist():

    print("Verifying existence of input files and directories...")
    critical_files_missing = False
    files_to_check = {
        "Speaker reference wav": SPEAKER_REFERENCE_WAV_PATH,
        "Checkpoint config file": CONFIG_CHECKPOINT_PATH,
        "XTTS checkpoint model": XTTS_CHECKPOINT_PATH,
        "XTTS tokenizer": TOKENIZER_PATH,
    }

    for description, path_obj in files_to_check.items():
        if not path_obj.exists():
            print(f"{description} not found at: {path_obj}")
            critical_files_missing = True
        else:
            print(f"Found: {description} at {path_obj}")


    if critical_files_missing:
        print("One or more input files are missing. Please check paths. Exiting.")
        sys.exit(1)
        
    print("All checked input files and directories exist.")




def main(text:str, language: str):

    check_input_files_exist()
    
    print("Initializing the XTTS model from configuration.")
    config = XttsConfig()
    config.load_json(CONFIG_CHECKPOINT_PATH)
    model = Xtts.init_from_config(config)
    
    print("Model loading from checkpoint...")
    model.load_checkpoint(
        config, 
        checkpoint_path=str(XTTS_CHECKPOINT_PATH), 
        vocab_path=str(TOKENIZER_PATH), 
        speaker_file_path=" ", 
        use_deepspeed=False,
        eval=True)
    
    print("Model loaded successfully.")

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[SPEAKER_REFERENCE_WAV_PATH],
        gpt_cond_len=30,
        gpt_cond_chunk_len=4,
        max_ref_length=60
    )


    print("Inference...") 
    out = model.inference(
        text= text,
        language= language,
        gpt_cond_latent= gpt_cond_latent,
        speaker_embedding= speaker_embedding,
        repetition_penalty= 5.0,
        temperature= 0.75,
    )

    torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    print(f"Output audio file '{OUTPUT_WAV_PATH}' saved")

if __name__ == "__main__":
    text = "cloud next iki bin yirmi beş etkinliği oldu."
    language = "tr"
    main(text, language)
