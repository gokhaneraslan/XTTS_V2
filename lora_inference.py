import sys
import os
import torch
import torchaudio
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from config import MAIN_DIR

from peft import PeftModel


SCRIPT_DIR = Path(MAIN_DIR + "/XTTS_V2")

PRETRAINED_MODEL_ROOT = SCRIPT_DIR / "pretrained_model"
OUTPUT_ROOT_DIR = SCRIPT_DIR / "training_output"
SPEAKER_REFERENCE_ROOT = SCRIPT_DIR / "speaker_reference"
OUTPUT_ROOT_DIR = SCRIPT_DIR / "training_output"


XTTS_MODEL_PATH = PRETRAINED_MODEL_ROOT / "model.pth"
XTTS_TOKENIZER_PATH = PRETRAINED_MODEL_ROOT / "vocab.json"


OUTPUT_PATH = OUTPUT_ROOT_DIR / "checkpoints/"
LORA_ADAPTER_PATH = OUTPUT_PATH / "lora_adapter"
XTTS_LORA_ORIGINAL_CONFIG_PATH = LORA_ADAPTER_PATH / "original_xtts_config.json"


SPEAKER_REFERENCE_WAV_PATH = SPEAKER_REFERENCE_ROOT / "reference.wav"
OUTPUT_PATH_WAV = OUTPUT_ROOT_DIR / "samples"
OUTPUT_WAV_PATH = OUTPUT_PATH_WAV / "output.wav"


def check_input_files_exist():

    print("Verifying existence of input files and directories...")
    critical_files_missing = False
    files_to_check = {
        "LORA adapter directory": LORA_ADAPTER_PATH,
        "Speaker reference wav": SPEAKER_REFERENCE_WAV_PATH,
        "original xtts config file": XTTS_LORA_ORIGINAL_CONFIG_PATH,
        "XTTS base model": XTTS_MODEL_PATH,
        "XTTS tokenizer": XTTS_TOKENIZER_PATH,
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
    config.load_json(XTTS_LORA_ORIGINAL_CONFIG_PATH)
    model = Xtts.init_from_config(config)
    
    print("Model loading from checkpoint...")
    model.load_checkpoint(
        config,
        checkpoint_path=str(XTTS_MODEL_PATH),
        vocab_path=str(XTTS_TOKENIZER_PATH),
        speaker_file_path=" ",
        use_deepspeed=False,
        eval=True)
    
    print("Model loaded successfully.")
    
    # During training, gpt inference layers are deleted. So we need to delete them here as well so that peft can load correctly.
    del model.gpt.gpt_inference
    
    print("Applying PEFT LoRA to the model...")
    model.gpt = PeftModel.from_pretrained(model.gpt, LORA_ADAPTER_PATH)

    print("Model peft loaded successfully!")

    if torch.cuda.is_available():
        model.cuda()
        
    # We are reinstalling the gpt inference layers because they were deleted
    model.gpt.init_gpt_for_inference()
    model.gpt.eval()
    
    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[SPEAKER_REFERENCE_WAV_PATH],
        gpt_cond_len=30,
        gpt_cond_chunk_len=4,
        max_ref_length=60
    )


    print("Inference...")
    out = model.inference(
        text = text,
        language= language,
        gpt_cond_latent= gpt_cond_latent,
        speaker_embedding= speaker_embedding,
        temperature=0.75,
    )

    os.makedirs(OUTPUT_PATH_WAV, exist_ok=True)
    torchaudio.save(str(OUTPUT_WAV_PATH), torch.tensor(out["wav"]).unsqueeze(0), 24000)
    print(f"Output audio file '{OUTPUT_WAV_PATH}' saved")


if __name__ == "__main__":
    text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
    language = "en"
    main(text, language)
