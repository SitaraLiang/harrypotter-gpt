# Harry Potter GPT

**A GPT-2 style Transformer trained from scratch to generate Harry Potter–inspired stories.**

This project implements a GPT-2–like decoder, trained on the complete Harry Potter book series. The model learns the style, tone, and vocabulary of the books, enabling it to generate new passages of text that feel like they came straight out of the wizarding world.


## **Features**

* **From-Scratch or Pretrained Initialization**

  * Train a GPT model from scratch.
  * Resume training from a checkpoint.

* **Flexible Tokenizers**

  * Supports character-level, BPE, SentencePiece, or tiktoken tokenization.
  * Easily swap tokenizers to experiment with sequence granularity and vocabulary size.

* **Training Features**

  * Gradient accumulation to simulate large batches.
  * Mixed precision training with `float16` or `bfloat16`.
  * Cosine learning rate schedule with warmup.
  * Gradient clipping for stable training.

* **Evaluation Metrics**

  * Standard **train/validation loss** estimation.

* **Checkpointing**

  * Automatic checkpoint saving on improved validation loss.
  * Full support for resuming training including model, optimizer, and counters.

* **Performance Optimizations**

  * Compatible with PyTorch 2.0 compilation (`torch.compile`) for faster training.
  * Efficient batch sampling using `np.memmap` for large datasets.
  * Model FLOPs utilization tracking for performance monitoring.


## **Setup**

1. Clone the repo:

```bash
git clone https://github.com/sitaraliang/harrypotter-gpt.git
cd harrypotter-gpt
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare dataset & tokenizer

   * Store tokenized training and validation sets as `train.bin` and `val.bin`.
   * Include a `meta.pkl` file with `vocab_size`.
   * For sentencepiece and bpe, we also store the corresponding tokenizer trained on our Harry Potter corpus.

4. (Optional) For GPU acceleration, ensure PyTorch with CUDA is installed.


## Dataset

For this project, I used a cleaned version of the full Harry Potter series from [Kaggle](https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books/data).

* The dataset contains seven `.txt` files corresponding to the seven books.
* Each book was cleaned to remove front pages and ending lines for easier NLP processing.

**Credit / Contact:**
Dataset originally provided by [smaindola90](mailto:smaindola90@gmail.com)

Using a simple command to tokenize and split into `train.bin` and `val.bin`.

For exampe, if you want to test the character-level tokenizer:
```bash
python data/harrypotter_char/prepare.py
```


## **Configuration**

Key parameters can be customized:

* `n_layer`, `n_head`, `n_embd` – Model size.
* `context_length` – Sequence length for training.
* `batch_size`, `gradient_accumulation_steps` – Batch configuration.
* `learning_rate`, `warmup_iters`, `lr_decay_iters` – Training schedule.
* `dropout` – Regularization.
* `device` – CPU, GPU, or MPS.


## Training

Train the model using the prepared dataset. For example: 

```sh
python train.py configs/train_bpe.py
```


* Based on the configuration, the model checkpoints are being written into the `--out_dir` directory respectively.
* Monitor training loss, training time and MFU during training.


## Text Generation / Sampling

Once the training finishes we can sample from the best model by pointing the sampling script at the checkpoints directory. For example: 

```sh
python sample.py --out_dir=out-harrypotter-tiktokenizer
```


## Future Work

* BLEU score evaluation on validation data to measure text generation quality, compare the performance of different tokenizers.
* Try parameter-efficient fine-tuning (LoRA / PEFT) larger models for higher-quality outputs
* Deploy as a web app on Hugging Face Spaces or Streamlit Cloud


## **Conclusion**

* Tokenizer choice indeed affects training dynamics and final performance.
* This project is also designed for experimentation: you can swap datasets, tokenizers, and model sizes easily.


