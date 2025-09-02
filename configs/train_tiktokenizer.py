# train a miniature character-level harry potter model
# good for debugging and playing on macbooks and such

out_dir = 'out-harrypotter-tiktokenizer' # where to save the checkpoints & logs
eval_interval = 250 # Run evaluation every 250 optimization steps.
                    # keep frequent because we'll overfit
eval_iters = 20 # # During each evaluation, run 200 minibatches to estimate metrics.
log_interval = 1 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

dataset = 'harrypotter_tiktokenizer'
gradient_accumulation_steps = 1
batch_size = 12
context_length = 256 

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.05

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'mps'
compile = False


"""
gradient_accumulation_steps = 1
batch_size = 12
context_length = 128
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.01
step 2000: train loss 3.5844, val loss 4.0452
"""

"""
gradient_accumulation_steps = 1
batch_size = 12
context_length = 256 
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.05
step 1750: train loss 3.3513, val loss 3.9973

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
"""