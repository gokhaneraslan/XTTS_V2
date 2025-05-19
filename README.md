# XTTS Fine-tuning Framework


This repository contains a framework for fine-tuning [Coqui-AI's TTS](https://github.com/coqui-ai/TTS) XTTS model, specialized for multilingual (with focus on Turkish) text-to-speech applications. It provides tools for both standard fine-tuning and LoRA (Low-Rank Adaptation) based fine-tuning.

## 📋 Overview

This project builds upon the XTTS model from Coqui-AI TTS to provide:

1. **Full Model Fine-tuning**: Traditional full parameter fine-tuning for XTTS
2. **LoRA Fine-tuning**: Memory-efficient fine-tuning using Low-Rank Adaptation
3. **Inference Scripts**: Tools to synthesize speech using both standard and LoRA-adapted models
4. **Support for Turkish**: Special focus on Turkish language support, but adaptable to other languages

## 🔧 Requirements & Setup

### Directory Structure

The project expects the following directory structure:

```
/
├── TTS/                   # Coqui-AI TTS library
├── pretrained_model/      # Contains original XTTS model files
│   ├── model.pth          # XTTS base model
│   ├── vocab.json         # XTTS tokenizer vocabulary
│   ├── dvae.pth           # Discrete VAE model
│   └── mel_stats.pth      # Mel spectrogram statistics
├── MyTTSDataset/          # Your custom dataset in LJSpeech format
│   ├── metadata.csv       # Dataset metadata
│   └── wavs/              # WAV files
│       └── segment_1.wav  # Reference audio sample
├── speaker_reference/     # Speaker reference audio files
│   └── segment_1.wav      # Reference audio for voice cloning
└── training_output/       # Training outputs
    ├── checkpoints/       # Saved model checkpoints
    └── samples/           # Generated audio samples
```

### Installation

1. Clone this repository
2. Install Coqui-AI TTS
```bash
pip install TTS
```
3. Install additional dependencies
```bash
pip install torch torchaudio peft
```

## 🚀 Usage

### 1. Preparing Your Dataset

Prepare your dataset in LJSpeech format:
- Audio files in 22050Hz, mono, WAV format
- `metadata.csv` with format: `file_name|raw text|normalized text`

### 2. Standard Fine-tuning

To fine-tune the entire XTTS model:

```bash
python train.py
```

### 3. LoRA Fine-tuning

To fine-tune using LoRA (more memory efficient):

```bash
python lora_train.py
```

### 4. Inference

#### Using Standard Fine-tuned Model

```bash
python inference.py
```

#### Using LoRA Fine-tuned Model

```bash
python lora_inference.py
```

#### Using LoRA with Training Framework

```bash
python lora_syntesize.py
```

## 📝 Script Descriptions

### `train.py`
Full model fine-tuning script. It loads the XTTS model and fine-tunes all parameters on your custom dataset.

### `lora_train.py`
LoRA-based fine-tuning script. It applies LoRA adapters to specific modules in the XTTS model, reducing memory requirements while still achieving good adaptation.

### `inference.py`
Inference script for the standard fine-tuned model. It loads a fully fine-tuned model and generates speech.

### `lora_inference.py`
Inference script for the LoRA fine-tuned model. It loads the base XTTS model and applies LoRA adapters for inference.

### `lora_syntesize.py`
Alternative inference script that utilizes the training framework for LoRA-based synthesis.

## ⚙️ Key Parameters

### Training Parameters

- **Model Configuration**:
  - `max_conditioning_length`: 132300
  - `min_conditioning_length`: 66150
  - `max_wav_length`: 255995
  - `max_text_length`: 200
  
- **Training Settings**:
  - `batch_size`: 4
  - `batch_group_size`: 48
  - `epochs`: 5 (LoRA) / 100 (Full)
  - `lr`: 1e-5 (LoRA) / 5e-6 (Full)

### LoRA Configuration

- `r`: 8 (LoRA rank)
- `lora_alpha`: 32
- `lora_dropout`: 0.05
- `target_modules`: ["c_attn", "c_proj", "c_fc"]

## 🔍 Important Considerations

1. **GPU Memory**: Full fine-tuning requires significant GPU memory (16GB+). LoRA reduces this requirement substantially.

2. **Training Time**: Expect training to take several hours to days depending on dataset size and hardware.

3. **Reference Audio**: The quality of the reference audio significantly impacts the voice cloning results.

4. **Language Support**: While focused on Turkish, the framework supports other languages supported by XTTS.

5. **File Paths**: Ensure all file paths are correctly set up before running the scripts.

## 📊 Performance Tips

1. **Speaker Reference**: Use high-quality, clean audio recordings for better voice cloning.

2. **Dataset Size**: For best results, use at least 30 minutes of transcribed speech data.

3. **Hyperparameters**: Adjust learning rate and batch size based on your dataset and hardware.

4. **Output Quality**: Test different temperature and repetition penalty settings during inference.

## 🎓 Technical Background

### Text-to-Speech (TTS)

Text-to-Speech systems convert written text into natural-sounding speech. Modern TTS systems typically use neural networks and can be categorized into:

1. **Concatenative TTS**: Combines pre-recorded speech segments
2. **Parametric TTS**: Synthesizes speech from parameters
3. **Neural TTS**: Uses deep learning to generate speech directly

### XTTS (Extended Text-to-Speech)

XTTS is Coqui-AI's advanced neural TTS model that supports:

- **Zero-shot voice cloning**: Mimicking voices from short audio samples
- **Multilingual synthesis**: Supporting multiple languages with a single model
- **Emotional speech**: Capturing emotional aspects of speech

### LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient fine-tuning technique that:

- Adds trainable low-rank matrices to frozen model weights
- Significantly reduces memory requirements during training
- Enables efficient adaptation of large language models

## 🙏 Acknowledgements

This project builds upon the excellent work of the [Coqui-AI TTS](https://github.com/coqui-ai/TTS) team. Special thanks to the original authors and contributors of the XTTS model and the TTS library.

## 📚 Further Reading

- [Coqui-AI TTS Documentation](https://tts.readthedocs.io/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [XTTS Technical Details](https://github.com/coqui-ai/TTS/tree/dev/TTS/tts/models/xtts)
