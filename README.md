# English-Vietnamese Machine Translation Project 🇻🇳🇺🇸

This repository contains the implementation of English-Vietnamese machine translation systems using two different approaches:
1.  **Fundamental Approach:** Training a Transformer model from scratch using PyTorch and Accelerate.
2.  **Modern Approach:** Fine-tuning a Large Language Model (Qwen2.5-0.5B) using LoRA and Quantization techniques for the medical domain.

## 💻 Environment & Hardware

* **Platform:** Kaggle Notebooks
* **GPU:** 2x NVIDIA Tesla T4 (2x 16GB VRAM)
* **Frameworks:** PyTorch, HuggingFace Transformers, PEFT, Accelerate, Evaluate.

---

## I. Part 1: Fundamentals (Transformer from Scratch)

In this section, we implemented a Transformer-based architecture to translate general English-Vietnamese text.

### 1. Dataset & Tokenizer
* **Dataset:** [ncduy/mt-en-vi](https://huggingface.co/datasets/ncduy/mt-en-vi)
* **Data Split:**
    * Training: 500,000 pairs
    * Validation: 11,000 pairs
    * Testing: 500 pairs
* **Tokenizer:** Custom tokenizer trained on the dataset with `vocab_size = 32000`.

### 2. Configuration
* **Architecture:** Transformer (Encoder-Decoder)
* **Positional Encoding:** Option to use Rotary Positional Embeddings (RoPE).
* **Decoding Strategy:** Beam Search (Beam Size = 3).
* **Training Strategy:** Two separate models were trained for En-Vi and Vi-En tasks.

### 3. Usage

#### Train Tokenizer
```bash
python tokenizer.py
# Example: Train with RoPE
```
#### Train model
```bash
accelerate launch main.py \
    --mode="train" \
    --use_rope="true" \
    --decode="beam" \
    --ckpt_name=None
```

#### Inference
```bash
accelerate launch main.py \
    --mode="inference" \
    --verbose="true" \
    --use_rope="true" \
    --decode="beam" \
    --ckpt_name="./checkpoints/your_checkpoint.pth"
```

#### Evaluate
```bash
accelerate launch main.py \
    --mode="evaluate" \
    --verbose="false" \
    --use_rope="true" \
    --decode="beam" \
    --ckpt_name="./checkpoints/your_checkpoint.pth"
```
### Result:
* **Eng to Vie: 29.74 Bleu score & 78.69 Comet score.**
* **Vie to Eng: 26.73 Bleu score & 76.63 Comet score.**
