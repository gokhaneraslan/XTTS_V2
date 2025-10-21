# 1. PATH SETTINGS
# ----------------------------------------------------------------
# Enter the absolute path to your project's main folder.
# Example: "/home/user/my_tts_project/"
MAIN_DIR = "/content"  # <<<--- PLEASE CHANGE THIS PATH TO YOURS

# Directory name for saving training outputs
OUTPUT_DIR_NAME = "training_output"


# 2. DATASET SETTINGS
# ----------------------------------------------------------------
# Directory name where your dataset is located
DATASET_DIR_NAME = "MyTTSDataset"

# Language code for fine-tuning ('en', 'tr', 'de', etc.)
LANGUAGE = "tr"
EVAL_SPLIT_RATIO = 0.1

# 3. TRAINING HYPERPARAMETERS
# ----------------------------------------------------------------
EPOCHS = 100
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 5e-06
GRAD_ACCUM_STEPS = 128  # Increase this if you run out of GPU memory.
SAVE_N_CHECKPOİNTS = 10

# 4. LOGGING & SAVING SETTINGS
# ----------------------------------------------------------------
# A name for your training run for Tensorboard or WandB
RUN_NAME = "xtts_finetune_v2"
PROJECT_NAME = "xtts_finetuning_project"

# A checkpoint will be saved every `save_step` steps.
SAVE_STEP = 1000
# Sample audio and plots will be sent to Tensorboard every `plot_step` steps.
PLOT_STEP = 100


# 5. TEST SENTENCES
# ----------------------------------------------------------------
# Sentences to be used to test the model's performance during training.
# The `speaker_wav` will be automatically set to the reference speaker audio defined in train.py.
TEST_SENTENCES = [
    {
        "text": "It took me quite a long time to develop a voice, and now that I have it I am not going to be silent.",
        "speaker_wav": None, # This will be filled automatically
        "language": "en",
    },
    {
        "text": "This cake is great. It's so delicious and moist.",
        "speaker_wav": None, # This will be filled automatically
        "language": "en",
    },
    {
        "text": "Bu, sesimi geliştirmemin oldukça uzun zamanımı aldı ve şimdi sahip olduğuma göre sessiz kalmayacağım.",
        "speaker_wav": None, # This will be filled automatically
        "language": "tr",
    },
    {
        "text": "Bu kek harika. Çok lezzetli ve nemli.",
        "speaker_wav": None, # This will be filled automatically
        "language": "tr",
    },
]