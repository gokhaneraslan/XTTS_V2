import sys
import os
import torch
import torchaudio
from pathlib import Path
from peft import PeftModel
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from config import MAIN_DIR


SCRIPT_DIR = Path(MAIN_DIR + "/XTTS_V2")

PRETRAINED_MODEL_ROOT = SCRIPT_DIR / "pretrained_model"
SPEAKER_REFERENCE_ROOT = SCRIPT_DIR / "speaker_reference"
OUTPUT_ROOT_DIR = SCRIPT_DIR / "training_output"


XTTS_MEL_PATH = PRETRAINED_MODEL_ROOT / "mel_stats.pth"
XTTS_DVAE_PATH = PRETRAINED_MODEL_ROOT / "dvae.pth"

XTTS_MODEL_PATH = PRETRAINED_MODEL_ROOT / "model.pth"
XTTS_TOKENIZER_PATH = PRETRAINED_MODEL_ROOT / "vocab.json"

SPEAKER_REFERENCE_WAV_PATH = SPEAKER_REFERENCE_ROOT / "reference.wav"
OUTPUT_PATH = OUTPUT_ROOT_DIR / "checkpoints/"

LORA_ADAPTER_PATH = OUTPUT_PATH / "lora_adapter"
OUTPUT_PATH = OUTPUT_ROOT_DIR / "samples"
OUTPUT_WAV_PATH = OUTPUT_ROOT_DIR / "output.wav"

def check_input_files_exist():

    print("Verifying existence of input files and directories...")
    critical_files_missing = False
    files_to_check = {
        "LORA adapter directory": LORA_ADAPTER_PATH,
        "Speaker reference wav": SPEAKER_REFERENCE_WAV_PATH,
        "XTTS Mel statistics": XTTS_MEL_PATH,
        "XTTS DVAE model": XTTS_DVAE_PATH,
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
    
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=255995,
        max_text_length=200,
        mel_norm_file=str(XTTS_MEL_PATH),
        dvae_checkpoint=str(XTTS_DVAE_PATH),
        xtts_checkpoint=str(XTTS_MODEL_PATH),
        tokenizer_file=str(XTTS_TOKENIZER_PATH),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000
    )


    config = GPTTrainerConfig(
        output_path=OUTPUT_PATH,
        model_args=model_args,
        run_name="xtts_finetune_v2",
        project_name="xtts_finetuning_project",
        run_description="Fine-tuning XTTS on LJSPEECH-like dataset",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=4,
        batch_group_size=48,
        eval_batch_size=2,
        num_loader_workers=4,
        eval_split_max_size=256,
        print_step=50,
        epochs=5,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=5,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=1e-5,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE_WAV_PATH,
                "language": "en",
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE_WAV_PATH,
                "language": "en",
            },
            {
                "text": "Bu, sesimi geliştirmemin oldukça uzun zamanımı aldı ve şimdi sahip olduğuma göre sessiz kalmayacağım.",
                "speaker_wav": SPEAKER_REFERENCE_WAV_PATH,
                "language": "tr",
            },
            {
                "text": "Bu kek harika. Çok lezzetli ve nemli.",
                "speaker_wav": SPEAKER_REFERENCE_WAV_PATH,
                "language": "tr",
            },
        ],
    )



    print("Initializing the GPTTrainer model from configuration.")
    model_container = GPTTrainer.init_from_config(config)
    print("Model initialized successfully.")
    
    print("Applying PEFT LoRA to the model...")
    model_container.xtts.gpt = PeftModel.from_pretrained(model_container.xtts.gpt, LORA_ADAPTER_PATH)

    print("Model peft loaded successfully!")

    if torch.cuda.is_available():
        model_container.cuda()

    model_container.xtts.gpt.init_gpt_for_inference() # if gpt_trainer delete the xtts.gpt.gpt_inference layers
    model_container.xtts.gpt.eval()
    
    print("Start to inference...")
    out = model_container.xtts.synthesize(
        text= text,
        config= config,
        speaker_wav= SPEAKER_REFERENCE_WAV_PATH,
        gpt_cond_len= 3,
        language= language,
    )

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    torchaudio.save(str(OUTPUT_WAV_PATH), torch.tensor(out["wav"]).unsqueeze(0), 24000)
    print(f"Output audio file '{OUTPUT_WAV_PATH}' saved")


if __name__ == "__main__":
    text="This cake is great. It's so delicious and moist."
    language = "en"
    main(text, language)
