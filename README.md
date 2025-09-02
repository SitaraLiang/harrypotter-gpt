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


## Installation

Clone the repo:

```bash
git clone https://github.com/sitaraliang/harrypotter-gpt.git
cd harrypotter-gpt
```

## **Setup**

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare dataset & tokenizer

   * Store tokenized training and validation sets as `train.bin` and `val.bin`.
   * Include a `meta.pkl` file with `vocab_size`.
   * For sentencepiece and bpe, we also store the corresponding tokenizer trained on our Harry Potter corpus.

3. (Optional) For GPU acceleration, ensure PyTorch with CUDA is installed.



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



## Training

Train the model using the prepared dataset. For example: 

```sh
python train.py configs/train_bpe.py
```


* Based on the configuration, the model checkpoints are being written into the `--out_dir` directory respectively.
* Monitor training loss, training time and MFU during training.


## **Configuration**

Key parameters can be customized:

* `n_layer`, `n_head`, `n_embd` – Model size.
* `context_length` – Sequence length for training.
* `batch_size`, `gradient_accumulation_steps` – Batch configuration.
* `learning_rate`, `warmup_iters`, `lr_decay_iters` – Training schedule.
* `dropout` – Regularization.
* `device` – CPU, GPU, or MPS.



## Text Generation / Sampling

Once the training finishes we can sample from the best model by pointing the sampling script at the checkpoints directory. For example: 

```sh
python sample.py --out_dir=out-harrypotter-tiktokenizer
```

This generates a few sample (I know they are not pretty but it's the best my macbook can do). For example:
```
“Do you, not?” said Dumbledore, emerging from beneath his nose. “They’ll have to hear you here, a book,” said Voldemort quietly. “You can fix anything to stop the name by the Dark Lord, or whatever else’s done them, they are merely the same.”

“What about Snape?” said Dumbledore, and Harry told them all. “Why did he have told me how to do he?”

“But that’s why he’s got the Horcrux, but he could not have sworn he’s gone, but he was too foolish to persuade Dumbledore, and -”

Harry did not believe his voice.

“What’s Dumbledore doing?” asked Dumbledore, “but he killed.”

“No, all he is right,” said Harry slowly. “You are very interested.”

“I can’t pretend it even it was because he was able to turn up to help the Horcrux, something terrible for the Ministry, seeing as we know better than I could, but told you what you were playing at school, but before he had ended up.”

“But if you think it was my scar,” said Dumbledore.

“I was telling you to ask him that Voldemort gets thinking that school,” he said quietly. “Harry Potter could do it, not, from the Death Eater. Dumbledore’s got to death, so I mean,” said Dumbledore. “For a minute?”

“It’s coming, Harry, I am not at Hogwarts, and can, you see that you sometimes have to kill,” said Dumbledore, still widening. “I am not sure of your father knew, but Harry, I think I was born in my office.”

Dumbledore nodded.

‘This is not a terrible wizard,” said Dumbledore, smiling, “and it is there with the Dark Lord. And who has he seen it. He never knew what the last time. But you had done, upon the subject. What is he?”

“I see, why go and have been lost in the orphanage.”

“Why?”

“This diary is as my word, sir?” said Dumbledore, with a sudden fear. He had left the house of Harry’s power. Snape.

“He’s not quite sure.”

“And so I knew that he was now nothing in my head, sir. All the Death Eaters are wrong at Hogwarts, and when he is.”

“And I am sure Dumbledore suffered older than death,” said Dumbledore. “And I see, so I’ve killed that. It did not return to Privet Drive, for it, and if he is here to come to Voldemort. And how you were?”

“I have no idea who I was born for Voldemort as Voldemort would have known that Voldemort will confide in Azkaban again. But the Ministry has been born in the darkness.”

“Yes,” said Harry, not to see him telling Dumbledore that his word was that Dumbledore had left for him. “He left me. He wanted to be more than you was?”

“I just told him that he was glad to ask you,” said Dumbledore, holding away. “Someone was a little more than most. And then they all work for all. I was a wizard who didn’t want to talk.”

“He wants to do it alone — we think of it was Death Eaters who had killed him. He was the school, but he was the one of them lived there. And then -“

“Yes, Harry, I know,” said Harry. “He would be aware of the time they were in the corridors.”

Dumbledore was listening to his mother at once; he didn’t know how to do anything to?”

“Yes,” said Harry, staring at his vision. “I don’t know where he was to kill you and Voldemort is still more than the castle: an old man, would they call him?

“He sees me for my own place. He sent me and nothing. He is too, sir.”

```


## Future Work

* BLEU score evaluation on validation data to measure text generation quality, compare the performance of different tokenizers.
* Try parameter-efficient fine-tuning (LoRA / PEFT) larger models for higher-quality outputs
* Deploy as a web app on Hugging Face Spaces or Streamlit Cloud


## **Conclusion**

* Tokenizer choice indeed affects training dynamics and final performance.
* This project is also designed for experimentation: you can swap datasets, tokenizers, and model sizes easily.


