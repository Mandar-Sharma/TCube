#!/usr/bin/env python
# coding: utf-8

import os
import datetime
import time
import random

import pandas as pd
import numpy as np
import re

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm

from transformers import (
	AdamW,
	GPT2LMHeadModel,
	GPT2Tokenizer, 
	GPT2Config,
	get_linear_schedule_with_warmup
)

class WebNLGNoSeq2Seq(Dataset):

	def __init__(self, tokenizer, max_source_length,
		max_target_length, type_path):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_source_length = max_source_length
		self.max_target_length = max_target_length
		self.input_ids = []
		self.attn_masks = []
		self._build(type_path)
	
	def __len__(self):
		return len(self.input_ids)
	
	def __getitem__(self, index):

		return self.input_ids[index], self.attn_masks[index]

	def _build(self, type_path):
		if type_path == 'train':
			df = pd.read_csv('Datasets/webnlg_train.csv')
		elif type_path == 'eval':
			df = pd.read_csv('Datasets/webnlg_dev.csv')
		else:
			df = pd.read_csv('Datasets/webnlg_test.csv')
			
		# n = 1 
		# df = df.head(int(len(df)*(n/100)))

		for index, row in df.iterrows():
				line = row['input_text']
				target = row['target_text']
				encodings = self.tokenizer('<|startoftext|>'+ line + ' = ' + target + '<|endoftext|>', truncation=True, max_length=self.max_source_length, padding="max_length")
				self.input_ids.append(torch.tensor(encodings['input_ids']))
				self.attn_masks.append(torch.tensor(encodings['attention_mask']))
				

class DARTNoSeq2Seq(Dataset):

	def __init__(self, tokenizer, max_source_length,
		max_target_length, type_path):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_source_length = max_source_length
		self.max_target_length = max_target_length
		self.input_ids = []
		self.attn_masks = []
		self._build(type_path)
	
	def __len__(self):
		return len(self.input_ids)
	
	def __getitem__(self, index):

		return self.input_ids[index], self.attn_masks[index]

	def _build(self, type_path):
		if type_path == 'train':
			df = pd.read_csv('Datasets/dart_train.csv')
		elif type_path == 'eval':
			df = pd.read_csv('Datasets/dart_dev.csv')
		else:
			df = pd.read_csv('Datasets/dart_test.csv')
			
		# n = 1 
		# df = df.head(int(len(df)*(n/100)))

		for index, row in df.iterrows():
				line = row['input_text']
				target = row['target_text']
				encodings = self.tokenizer('<|startoftext|>'+ line + ' = ' + target + '<|endoftext|>', truncation=True, max_length=self.max_source_length, padding="max_length")
				self.input_ids.append(torch.tensor(encodings['input_ids']))
				self.attn_masks.append(torch.tensor(encodings['attention_mask']))


class BOTHNoSeq2Seq(Dataset):

	def __init__(self, tokenizer, max_source_length,
		max_target_length, type_path):
		super().__init__()
		self.tokenizer = tokenizer
		self.max_source_length = max_source_length
		self.max_target_length = max_target_length
		self.input_ids = []
		self.attn_masks = []
		self._build(type_path)
	
	def __len__(self):
		return len(self.input_ids)
	
	def __getitem__(self, index):

		return self.input_ids[index], self.attn_masks[index]

	def _build(self, type_path):
		if type_path == 'train':
			df1 = pd.read_csv('Datasets/dart_train.csv')
			df2 = pd.read_csv('Datasets/webnlg_train.csv')
		elif type_path == 'eval':
			df1 = pd.read_csv('Datasets/dart_dev.csv')
			df2 = pd.read_csv('Datasets/webnlg_dev.csv')
		else:
			df1 = pd.read_csv('Datasets/dart_test.csv')
			df2 = pd.read_csv('Datasets/webnlg_test.csv')
		
		df = pd.concat([df1, df2])
		# n = 1 
		# df = df.head(int(len(df)*(n/100)))

		for index, row in df.iterrows():
				line = row['input_text']
				target = row['target_text']
				encodings = self.tokenizer('<|startoftext|>'+ line + ' = ' + target + '<|endoftext|>', truncation=True, max_length=self.max_source_length, padding="max_length")
				self.input_ids.append(torch.tensor(encodings['input_ids']))
				self.attn_masks.append(torch.tensor(encodings['attention_mask']))


tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
print("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))

batch_size = 4
train_dataset = BOTHNoSeq2Seq(tokenizer, 512, 512, 'train')
val_dataset = BOTHNoSeq2Seq(tokenizer, 512, 512, 'val')
test_dataset = BOTHNoSeq2Seq(tokenizer, 512, 512, 'test')

train_dataloader = DataLoader(
			train_dataset,  # The training samples.
			sampler = RandomSampler(train_dataset), # Select batches randomly
			batch_size = batch_size # Trains with this batch size.
		)

validation_dataloader = DataLoader(
			val_dataset, # The validation samples.
			sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
			batch_size = batch_size # Evaluate with this batch size.
		)
test_dataloader = DataLoader(
			test_dataset, # The validation samples.
			sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
			batch_size = batch_size # Evaluate with this batch size.
		)

configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

model.resize_token_embeddings(len(tokenizer))

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda:1")
model.cuda()

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

epochs = 5
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

# this produces sample output every 100 steps
sample_every = 1000

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(model.parameters(),
				  lr = learning_rate,
				  eps = epsilon
				)

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, 
											num_warmup_steps = warmup_steps, 
											num_training_steps = total_steps)

def format_time(elapsed):
	return str(datetime.timedelta(seconds=int(round((elapsed)))))

total_t0 = time.time()

training_stats = []

model = model.to(device)

for epoch_i in range(0, epochs):

	# ========================================
	#               Training
	# ========================================

	print("")
	print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
	print('Training...')

	t0 = time.time()

	total_train_loss = 0

	model.train()

	for step, batch in tqdm(enumerate(train_dataloader)):

		b_input_ids = batch[0].to(device)
		b_labels = batch[0].to(device)
		b_masks = batch[1].to(device)

		model.zero_grad()        

		outputs = model(  b_input_ids,
						  labels=b_labels, 
						  attention_mask = b_masks,
						  token_type_ids=None
						)

		loss = outputs[0]  

		batch_loss = loss.item()
		total_train_loss += batch_loss

		# Get sample every x batches.
		if step % sample_every == 0 and not step == 0:

			elapsed = format_time(time.time() - t0)
			print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

			model.eval()

			sample_outputs = model.generate(
									bos_token_id=random.randint(1,30000),
									do_sample=True,   
									top_k=50, 
									max_length = 200,
									top_p=0.95, 
									num_return_sequences=1
								)
			for i, sample_output in enumerate(sample_outputs):
				  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
			
			model.train()

		loss.backward()

		optimizer.step()

		scheduler.step()

	# Calculate the average loss over all of the batches.
	avg_train_loss = total_train_loss / len(train_dataloader)       
	
	# Measure how long this epoch took.
	training_time = format_time(time.time() - t0)

	print("")
	print("  Average training loss: {0:.2f}".format(avg_train_loss))
	print("  Training epoch took: {:}".format(training_time))
		
	# ========================================
	#               Validation
	# ========================================

	print("")
	print("Running Validation...")

	t0 = time.time()

	model.eval()

	total_eval_loss = 0
	nb_eval_steps = 0

	# Evaluate data for one epoch
	for batch in tqdm(validation_dataloader):
		
		b_input_ids = batch[0].to(device)
		b_labels = batch[0].to(device)
		b_masks = batch[1].to(device)
		
		with torch.no_grad():        

			outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
							 attention_mask = b_masks,
							labels=b_labels)
		  
			loss = outputs[0]  
			
		batch_loss = loss.item()
		total_eval_loss += batch_loss        

	avg_val_loss = total_eval_loss / len(validation_dataloader)
	
	validation_time = format_time(time.time() - t0)    

	print("  Validation Loss: {0:.2f}".format(avg_val_loss))
	print("  Validation took: {:}".format(validation_time))

	# Record all statistics from this epoch.
	training_stats.append(
		{
			'epoch': epoch_i + 1,
			'Training Loss': avg_train_loss,
			'Valid. Loss': avg_val_loss,
			'Training Time': training_time,
			'Validation Time': validation_time
		}
	)

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

output_dir = './outputs/GPT2BOTH/'
output_dir_bu = './outputs_bu/GPT2BOTH/'

# Create output directory if needed
if not os.path.exists(output_dir_bu):
	os.makedirs(output_dir_bu)

torch.save(model.state_dict(), output_dir_bu + 'GPT2BOTH.bin')

# Create output directory if needed
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)



# """# Generate Text"""

# model.eval()

# prompt = "<|startoftext|>"

# generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
# generated = generated.to(device)

# print(generated)

# sample_outputs = model.generate(
#                                 generated, 
#                                 #bos_token_id=random.randint(1,30000),
#                                 do_sample=True,   
#                                 top_k=50, 
#                                 max_length = 300,
#                                 top_p=0.95, 
#                                 num_return_sequences=3
#                                 )

# for i, sample_output in enumerate(sample_outputs):
#   print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))