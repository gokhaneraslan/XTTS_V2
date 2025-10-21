import sys
from pathlib import Path

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

import config


MAIN_PATH = Path(config.MAIN_DIR)
SCRIPT_DIR = MAIN_PATH / "XTTS_V2"

PRETRAINED_MODEL_ROOT = SCRIPT_DIR / "pretrained_model"
DATASET_ROOT = SCRIPT_DIR / config.DATASET_DIR_NAME
SPEAKER_REFERENCE_ROOT = SCRIPT_DIR / "speaker_reference"
OUTPUT_PATH = SCRIPT_DIR / config.OUTPUT_DIR_NAME


XTTS_MEL_PATH = PRETRAINED_MODEL_ROOT / "mel_stats.pth"
XTTS_DVAE_PATH = PRETRAINED_MODEL_ROOT / "dvae.pth"
XTTS_MODEL_PATH = PRETRAINED_MODEL_ROOT / "model.pth"
XTTS_TOKENIZER_PATH = PRETRAINED_MODEL_ROOT / "vocab.json"
METADATA_PATH = DATASET_ROOT / "metadata.csv"
SPEAKER_REFERENCE_WAV_PATH = str(SPEAKER_REFERENCE_ROOT / "reference.wav")


def check_input_files_exist():
    """Checks if all required input files and directories exist."""
    print("Checking for the existence of input files and directories...")
    files_to_check = {
        "Speaker reference wav": Path(SPEAKER_REFERENCE_WAV_PATH),
        "Dataset directory": DATASET_ROOT,
        "XTTS Mel statistics": XTTS_MEL_PATH,
        "XTTS DVAE model": XTTS_DVAE_PATH,
        "XTTS base model": XTTS_MODEL_PATH,
        "XTTS tokenizer": XTTS_TOKENIZER_PATH,
        "Metadata file": METADATA_PATH,
    }

    if not all(path.exists() for path in files_to_check.values()):
        for name, path in files_to_check.items():
            if not path.exists():
                print(f"ERROR: {name} not found at -> {path}")
        print("\nMissing files detected. Please check the paths in config.py and file names. Exiting.")
        sys.exit(1)

    print("All input files and directories exist.")


def main():
    """Main training function."""
    check_input_files_exist()

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name=config.DATASET_DIR_NAME,
        path=str(DATASET_ROOT),
        meta_file_train=str(METADATA_PATH),
        language=config.LANGUAGE,
    )


    train_samples, eval_samples = load_tts_samples(
        [dataset_config], 
        eval_split=True, 
        eval_split_max_size=256, 
        eval_split_size=config.EVAL_SPLIT_RATIO
    )
    print(f"\nLoaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")


    audio_config = XttsAudioConfig(
        sample_rate=22050, 
        dvae_sample_rate=22050, 
        output_sample_rate=24000
    )
    
    model_args = GPTArgs(
        max_conditioning_length=132300, 
        min_conditioning_length=66150,
        max_wav_length=255995, 
        max_text_length=512,
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


    for sentence in config.TEST_SENTENCES:
        if sentence.get("speaker_wav") is None:
            sentence["speaker_wav"] = SPEAKER_REFERENCE_WAV_PATH

    trainer_config = GPTTrainerConfig(
        output_path=str(OUTPUT_PATH),
        model_args=model_args,
        run_name=config.RUN_NAME,
        project_name=config.PROJECT_NAME,
        run_description="Fine-tuning XTTS v2 model.",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=config.BATCH_SIZE,
        eval_batch_size=config.EVAL_BATCH_SIZE,
        num_loader_workers=4,
        print_step=50,
        plot_step=config.PLOT_STEP,
        log_model_step=1000,
        save_step=config.SAVE_STEP,
        epochs=config.EPOCHS,
        save_n_checkpoints=config.SAVE_N_CHECKPOİNTS,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=config.LEARNING_RATE,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 150000, 300000], "gamma": 0.5},
        test_sentences=config.TEST_SENTENCES,
    )

    trainer_args = TrainerArgs(
        grad_accum_steps=config.GRAD_ACCUM_STEPS,
        start_with_eval=True,
    )


    print("\nInitializing the GPTTrainer model from configuration...")
    model = GPTTrainer.init_from_config(trainer_config)
    
    trainer = Trainer(
        args=trainer_args,
        config=trainer_config,
        output_path=str(OUTPUT_PATH),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    print("\nTrainer initialized successfully. Starting training... 🚀")
    trainer.fit()
    print("\n------- Training Finished Successfully ----------")

if __name__ == "__main__":
    main()