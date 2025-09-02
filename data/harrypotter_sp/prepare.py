"""
Train a custom SentencePiece tokenizer.
"""
import os
import sentencepiece as spm
from pathlib import Path
import numpy as np
import pickle

base_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(base_dir)
input_dir = Path(os.path.join(parent_dir, 'clean'))
input_file_path = input_dir / "input.txt"

# Reads entire harry potter series into memory as a single string data.
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read() # data: a single string

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

options = dict(
  # input spec
  input=str(input_file_path),
  input_format="text",
  # output spec
  model_prefix=os.path.join(base_dir, "tok5000"),
  # algorithm spec
  # BPE alg
  model_type="bpe",
  vocab_size=5000,
  # normalization
  normalization_rule_name="identity", # ew, turn off normalization
  remove_extra_whitespaces=False,
  input_sentence_size=200000000, # max number of training sentences
  max_sentence_length=4192, # max number of bytes per sentence
  seed_sentencepiece_size=1000000,
  shuffle_input_sentence=True,
  # rare word treatment
  character_coverage=0.99995,
  byte_fallback=True,
  # merge rules
  split_digits=True,
  split_by_unicode_script=True,
  split_by_whitespace=True,
  split_by_number=True,
  max_sentencepiece_length=16,
  add_dummy_prefix=True,
  allow_whitespace_only_pieces=True,
  # special tokens
  unk_id=0, # the UNK token MUST exist
  bos_id=1, # the others are optional, set to -1 to turn off
  eos_id=2,
  pad_id=-1,
  # systems
  num_threads=os.cpu_count(), # use ~all system resources
)

spm.SentencePieceTrainer.train(**options)

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(base_dir, 'tok5000.model'))
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]

# Encode to token IDs
train_ids = sp.encode(train_data, out_type=int)
val_ids = sp.encode(val_data, out_type=int)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(base_dir, 'train.bin'))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

vocab_file = os.path.join(base_dir, 'meta.pkl')
meta = {
    'vocab_size': sp.get_piece_size(),
}

print(f"vocab size: {sp.get_piece_size()}")

with open(vocab_file, 'wb') as f:
    pickle.dump(meta, f)


"""
train has 1,526,476 tokens
val has 169,552 tokens
vocab size: 5000
""" 

"""
step 2000: train loss 3.4342, val loss 4.0423


Though he said he had thought he had heard footsteps as a glass back door once more; he seemed to crash a loud crack.

“Had on, that's more, you know, Harry! But now we lost-magical, and we've got an owl's not as much as you can get a work on the school! Ron and Hermione will have to wait for the match!”

Harry could almost move to the Triwizard Tournament; it was a real bit of fun: Colin Dolores Jinx Eight Cross, where he was going to take off up to the Gryffindor common room, and the following day to be seen.

“I haven't been in Divination,” said Hermione, who was still glaring over the twins' table. “We'd better get on, we’re changing at the hospital wing. Harry, we'll both finish the Snitch, Dumbledore! We shall be in there borrow evening and tell us what we know about about the talicose.”

“You feel have more like this,” said Harry, “it's got any of them again. I'm not sure I'm going to do something?”

“I don't know anything about the team,” said Ron, who was still more annoyed. “I was being a lot of time if you don't know who the only person I might be back in the castle before the front door with you. I can learn of it.”

“I'm nearly warned,” said Harry. “We found it in the forest.”

“Yeah, still now, well,” said Ron, “he’s got enough, a bit better.”

“Good afternoon, Lavender,” said Harry.

“Yeah, well,” said Hermione, pointing at Ron, frowning. “You can go out with that, but they don’t mind for anything wrong. I gave any wizards if Dad says you were any of you. Harry, really, but you thought she had a quick deal of order with a very good, they’ve got to become a bit. I’d like a term, and I’ll be able to see some Captain. It knows you’ve been really horrible. I’d better go off. There’s no need to be some sort of magic in magic. The only person has done it out.”

“That’s a very bad one of the other end, isn’t it?” said Malfoy, as though Harry had pelted dicting with his wand.

“We’re not here,” said Harry, “you’re not alone — we shouldn’t have been sending you to your aunt and uncle.”

“What’s the —’”

“I’ll take you the Snitch!” said Malfoy, who had been in the Gryffindor common room, and Harry imagined it, but there was a horrible feeling that the rest of the team players had had a rather shuffling along a street.

“We’re in there, please — we’ve got to eat at Hogwarts,” said Malfoy, closing the door behind Malfoy, who was already faded off for a glimpse of the Selow game. “Arechie, Harry, come — you know, Horce.”

“This was the one thing they’d better get up,” snapped Wood.

“Ginny, Mr. Malfoy didn’t return to the school anymore, but it was only a crime. Hornky,” and turning to Harry. “That’s not my fault.”

“C’mon, Percy and George’s just getting bidden from the feast in a cauldrons. There would have a few minutes to find out who would be able to share your team over Hogwarts Creevey. …”

“Kreacher thinks’s the Galleons for the whole school will. When he’s nearly last time, I’m not having a lot more school to survive to take an Felix Felicis.”

Harry was sure there was a handsome, the Committee. There was a lot of people who had not told them that he was finding the sign of the Dark Arts. He had never been seen, and the rest of the team presented.

“I didn’t want to be glad to make his way,” said Riddle. “Hang on, Harry.”

“I’ve not been listening to you, not to think about what to think about a good boy,” he added, in a low voice, “thill I can’t get much longer out in the hospital wing. You’d be able to use it.”





The crowd seemed to hold. Harry's, however, was feeling that he was trying to keep himself on the back of his mind, but he had never been better than Occlumency, he would be in the library; he was sure he was not going to prove something.

'I'm keeping him down here?' said Ron, trying to suppress eye now. 'But I'm not going on in trouble. I'm feeling it's been any accidents when it's ever found out the place in this.'

Hermione cast a pheg in a kind of slut of parchment from her nose so that there should be some sort of thing. She led her up to the right of the lesson.

'Wate a stack of metalad and advenetive Defence Against the Dark Arts teacher, and I'm going to be in there and not to stay at school, not a good idea of what we’re doing, he can have to know her!'

'I'm not sure,' said Luna, sitting up at the end of the lesson, ignoring the same way. 'Have you got any more, why aren't you?'

He looked around, but Harry did not think that she was not sure, but -

'Oooh!' said Angelina, sitting down, very much back outside. 'Course we can tell me?'

'No,' said Harry, staring at the same jar and they were all looking anxiously at the chains. 'Well… but will you see? What, please?'

Harry felt Hermione and Ron were still but tore a greater, his eyes still whirl-like. He had only thought it had not been he'd had to take the letter on him; the whole rows of broken-to-lined hooke was dashing and swinging it over either side of the room.

'You'd better get going upstairs, that's not there tomorrow?'

'We're being so scared about something I was worried to you?' asked Ron.

'Yeah,' said Ron, stroking Harry. 'You can't tell him? I'll bet the truth. He'll just have that egg, isn't he, you're not doing, Harry. They find my mother's family, doesn't he? Who d'you mean?'

'Don't you think he might have been stupid,' said Ron, 'yeah, I can't tell us what?'

'Luna - but - but she's just got some months at the Ministry. I've got the stupid thing ter do, I've been in the room because you're haffing to the ball in here. I was too much of any more.'

She flicked her hand, glared at Hermione, her foot of her mouth.

'Whatas you ever said to the third floor again, 'I'm up with me!'

But Ron and Hermione nodded, but Snape said, 'He's obviously not going to be, Harry's' after Neak on his back -'

'She's still, there's no way of us, Molly,' said Hermione, whose hands on her and Luna was watching him in a charge or two. 'You've got a friend of yourselves with me, Potter, after that. Good-old an appointment,' said Ron bitterly. 'It was from the ball…'

'Good,' said Hermione, sounding up. Harry was sure he knew what to is, he said, 'I've got to tell him if you want to know how that we've won an owl, you wanted to.'

'How're we going to say. It's not easy to defend the school of the Order?' asked Ginny, who was holding the door shut on Hermione and Hermione's ear.

'And that's Dumbledore, what's going on?' said Harry.

'We're going to find out where Told?' said Ginny, looking anxious.

'You think we're going to bother for that,' said Ron. 'I dunno how many people say that I've had to do an owl. They're doing, have you?'

'Not in a week, I want to know where I've found the truth. They're all right, anyway.'

'No, I've been thinking about what they have to do. That's the matter I'm afraid you're inhin'?'

'I'm not,' said Hermione. 'And then, I said you'd better get


"""