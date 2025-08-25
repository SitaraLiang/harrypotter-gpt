# 🪄 Harry Potter GPT

**Generate Harry Potter–style stories using a custom-trained GPT model.**

This project fine-tunes a GPT-style Transformer on the full Harry Potter book series and provides an interactive demo for generating fanfic-style text. It may further include a “from-scratch” GPT implementation for learning purposes.



## 📂 Project Structure

```
harrypotter-gpt/
├── data/                 # Cleaned corpus text, tokenized train/val data
│   ├── harrypotter_clean/         # Cleaned corpus text
│   ├── harrypotter_char/
|   │   ├── prepare.py  
│   │   └── readme.md   
│   └── harrypotter/
|       ├── prepare.py  
│       └── readme.md   
 
├── src/                  # Source code adapted from nanoGPT (model, dataset, training, sampling)
│   ├── model.py           
│   ├── train.py
│   └── sample.py      
├── demo/                 # Interactive apps
│   ├── gradio_app.py
│   └── streamlit_app.py
├── notebooks/            # Exploratory notebooks
├── configs/              # Training and model configs
├── checkpoints/          # Saved model weights
├── requirements.txt
└── README.md
```



## ⚡ Features

* Fine-tune a GPT model on the Harry Potter corpus
* Generate stories in Harry Potter style
* Interactive demo via Gradio (or Streamlit)
* Optional from-scratch GPT implementation for learning



## 🛠️ Installation

1. Clone the repo:

```bash
git clone https://github.com/sitaraliang/harrypotter-gpt.git
cd harrypotter-gpt
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) For GPU acceleration, ensure PyTorch with CUDA is installed.



## 📝 Dataset

For this project, I used a cleaned version of the full Harry Potter series from [Kaggle](https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books/data).

* The dataset contains 7 `.txt` files corresponding to the 7 books.
* Each book was cleaned to remove front pages and ending lines for easier NLP processing.

**Credit / Contact:**
Dataset originally provided by [smaindola90](mailto:smaindola90@gmail.com)

Using a simple command to tokenize and split into `train.bin` and `val.bin`.

Character-level tokenizer:
```bash
python data/harrypotter_char/prepare.py
```

Or

GPT-2 BPE tokenizer:
```bash
python data/harrypotter/prepare.py
```



## 🚀 Training

Train the model using the prepared dataset:

```sh
python train.py configs/train_harrypotter_char.py
```

Or

```sh
python train.py config/train_harrypotter.py
```

* Checkpoints are saved in `checkpoints/`.
* Monitor training loss and generate sample text during training.



## 🎨 Text Generation / Sampling

Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-harrypotter-char
```

Or

```sh
python sample.py --out_dir=out-harrypotter
```

This generates a few samples, for example:



## 🌐 Interactive Demo

Run the Gradio app:

```bash
python demo/gradio_app.py
```

* Enter a story prompt, and the model will continue it in Harry Potter style.
* Optional: Run `streamlit_app.py` for a Streamlit dashboard.



## 🏆 Future Work

* Experiment with character-level vs BPE tokenization
* Build a Tokenizer from scratch
* Deploy as a web app on Hugging Face Spaces or Streamlit Cloud
* Fine-tune larger models for higher-quality outputs
* Try parameter-efficient fine-tuning (LoRA / PEFT)
* Build a from-scratch GPT implementation
* Compare outputs between from-scratch GPT and nanoGPT


