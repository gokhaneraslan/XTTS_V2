from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from pathlib import Path
import sys
import os
from peft import LoraConfig, get_peft_model
from config import MAIN_DIR

SCRIPT_DIR = Path(MAIN_DIR + "/XTTS_V2")

PRETRAINED_MODEL_ROOT = SCRIPT_DIR / "pretrained_model"
MYTTSDATASET_ROOT = SCRIPT_DIR / "MyTTSDataset"
SPEAKER_REFERENCE_ROOT = SCRIPT_DIR / "speaker_reference"
OUTPUT_ROOT_DIR = SCRIPT_DIR / "training_output"


XTTS_MEL_PATH = PRETRAINED_MODEL_ROOT / "mel_stats.pth"
XTTS_DVAE_PATH = PRETRAINED_MODEL_ROOT / "dvae.pth"
XTTS_MODEL_PATH = PRETRAINED_MODEL_ROOT / "model.pth"
XTTS_TOKENIZER_PATH = PRETRAINED_MODEL_ROOT / "vocab.json"


SPEAKER_REFERENCE_WAV_PATH = SPEAKER_REFERENCE_ROOT / "reference.wav"
OUTPUT_PATH = OUTPUT_ROOT_DIR / "checkpoints/"

LORA_ADAPTER_PATH = OUTPUT_PATH / "lora_adapter"
XTTS_LORA_ORIGINAL_CONFIG_PATH = LORA_ADAPTER_PATH / "original_xtts_config.json"



config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path= str(MYTTSDATASET_ROOT),
    meta_file_train= str(MYTTSDATASET_ROOT / "metadata.csv"),
    language="tr",
)

DATASETS_CONFIG_LIST = [config_dataset]
LANGUAGE = config_dataset.language

def check_input_files_exist():

    print("Verifying existence of input files and directories...")
    critical_files_missing = False
    files_to_check = {
        "Speaker reference wav": SPEAKER_REFERENCE_WAV_PATH,
        "Dataset directory": MYTTSDATASET_ROOT,
        "XTTS Mel statistics": XTTS_MEL_PATH,
        "XTTS DVAE model": XTTS_DVAE_PATH,
        "XTTS base model": XTTS_MODEL_PATH,
        "XTTS tokenizer": XTTS_TOKENIZER_PATH,
        "Metadata file": MYTTSDATASET_ROOT / "metadata.csv",
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



def main():

    check_input_files_exist()

    print("Loading training and evaluation samples...")
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=256,
        eval_split_size=0.05,
    )

    print(f"Loaded {len(train_samples) if train_samples else 0} training samples and {len(eval_samples) if eval_samples else 0} evaluation samples.")


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
        output_path=str(OUTPUT_PATH),
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
        log_model_step=1000,
        save_step=10000,
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
                "speaker_wav": str(SPEAKER_REFERENCE_WAV_PATH),
                "language": "en",
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": str(SPEAKER_REFERENCE_WAV_PATH),
                "language": "en",
            },
            {
                "text": "Bu, sesimi geliştirmemin oldukça uzun zamanımı aldı ve şimdi sahip olduğuma göre sessiz kalmayacağım.",
                "speaker_wav": str(SPEAKER_REFERENCE_WAV_PATH),
                "language": "tr",
            },
            {
                "text": "Bu kek harika. Çok lezzetli ve nemli.",
                "speaker_wav": str(SPEAKER_REFERENCE_WAV_PATH),
                "language": "tr",
            },
        ],
    )


    print("Initializing the GPTTrainer model from configuration.")
    model_peft = GPTTrainer.init_from_config(config)
    
    print("Model initialized successfully.")
    
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["c_attn", "c_proj", "c_fc"],
        #target_modules = ["c_attn", "c_proj", "c_fc", "qkv", "proj_out", "to_q", "to_kv", "to_out"],
    )

    print("Applying PEFT LoRA to the model...")
    model_peft.xtts.gpt = get_peft_model(model_peft.xtts.gpt, peft_config)

    trainable_params, total_params = model_peft.xtts.gpt.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params}")


    trainer_args = TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=True,
        grad_accum_steps=128,
    )
    
    
    print("Trainer initialized successfully. Starting training... 🚀")
    trainer = Trainer(
        args=trainer_args,
        config=config,
        output_path=str(OUTPUT_PATH),
        model=model_peft,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )


    trainer.fit()

    print("------- Training Finished Successfully ----------")
    
    print("Saving LORA...")
    os.makedirs(LORA_ADAPTER_PATH, exist_ok=True)
    model_peft.xtts.gpt.save_pretrained(str(LORA_ADAPTER_PATH))
    config.save_json(str(XTTS_LORA_ORIGINAL_CONFIG_PATH))

    print(f"LoRA '{LORA_ADAPTER_PATH}' saved")
    print(f"Original XTTS config '{XTTS_LORA_ORIGINAL_CONFIG_PATH}' saved.")


if __name__ == "__main__":
    main()
