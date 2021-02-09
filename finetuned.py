import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartTokenizer, BartForConditionalGeneration

cuda_default = torch.device("cuda:0")

class T5FineTuner(pl.LightningModule):
	def __init__(self):
		super(T5FineTuner, self).__init__()
		self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
		self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

		new_tokens = ['<H>', '<R>', '<T>']
		new_tokens_vocab = {}
		new_tokens_vocab['additional_special_tokens'] = []
		for idx, t in enumerate(new_tokens):
			new_tokens_vocab['additional_special_tokens'].append(t)
		self.tokenizer.add_special_tokens(new_tokens_vocab)
		self.model.resize_token_embeddings(len(self.tokenizer))

class BARTFineTuner(pl.LightningModule):
	def __init__(self, hparams):
		super(BARTFineTuner, self).__init__()
		self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
		self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
		new_tokens = ['<H>', '<R>', '<T>']
		new_tokens_vocab = {}
		new_tokens_vocab['additional_special_tokens'] = []
		for idx, t in enumerate(new_tokens):
			new_tokens_vocab['additional_special_tokens'].append(t)
		self.tokenizer.add_special_tokens(new_tokens_vocab)
		self.model.resize_token_embeddings(len(self.tokenizer))

def generate(model, text, prefix = "", cuda_dev=cuda_default, max_len=512):
	model.eval()
	input_ids = model.tokenizer.encode(prefix + text, return_tensors="pt").to(cuda_dev)
	outputs = model.model.generate(input_ids, max_length=max_len)
	return model.tokenizer.decode(outputs[0])

def generate_ngram_es(model, text, prefix = "", ngram = 1, early = True, cuda_dev = cuda_default, max_len=512):
	model.eval()
	input_ids = model.tokenizer.encode(prefix + text, return_tensors="pt").to(cuda_dev)
	outputs = model.model.generate(input_ids, no_repeat_ngram_size = ngram, max_length=max_len, early_stopping=early)
	return model.tokenizer.decode(outputs[0])

def generate_topk(model, text, prefix = "", topk = 50, cuda_dev = cuda_default, max_len=512):
	model.eval()
	input_ids = model.tokenizer.encode(prefix + text, return_tensors="pt").to(cuda_dev)
	outputs = model.model.generate(input_ids, do_sample=True, max_length=max_len, top_k=topk)
	return model.tokenizer.decode(outputs[0])

def generate_topp(model, text, prefix = "", topp = 0.92, cuda_dev = cuda_default, max_len=512):
	model.eval()
	input_ids = model.tokenizer.encode(prefix + text, return_tensors="pt").to(cuda_dev)
	outputs = model.model.generate(input_ids, do_sample=True, max_length=max_len, top_p=topp, top_k=0)
	return model.tokenizer.decode(outputs[0])


def generate_beam(model, text, prefix = "", max_len=512, n_beam = 5, ngram = 1, r_seq = 5, early = True, cuda_dev = cuda_default):
	model.eval()
	input_ids = model.tokenizer.encode(prefix + text, return_tensors="pt").to(cuda_dev)
	outputs = model.model.generate(input_ids, num_beams = n_beam, no_repeat_ngram_size = ngram, num_return_sequences=r_seq, max_length=max_len, early_stopping=early)
	for i, beam_output in enumerate(outputs):
		print("{}: {}".format(i, model.tokenizer.decode(beam_output, skip_special_tokens=True)))
		
def graph2text_nobeam(model, graph, prefix = "", max_len=512, cuda_dev=cuda_default):
	output = ""
	for entry in graph:
		text = generate(model, entry, prefix, cuda_dev, max_len=512)
		output = output + text +  ' '
	return output

def graph2text_nobeam_ngram_es(model, graph, prefix = "", ngram = 1, early = True, max_len=512,  cuda_dev=cuda_default):
	output = ""
	for entry in graph:
		text = generate_ngram_es(model, entry, prefix, ngram, early, cuda_dev, max_len=max_len)
		output = output + text +  ' '
	return output

def graph2text_nobeam_topk(model, graph, prefix = "", topk = 50, max_len=512,  cuda_dev=cuda_default):
	output = ""
	for entry in graph:
		text = generate_topk(model, entry, prefix, topk, cuda_dev, max_len=max_len)
		output = output + text +  ' '
	return output

def graph2text_nobeam_topp(model, graph, prefix = "", topp = 0.92, max_len=512, cuda_dev=cuda_default):
	output = ""
	for entry in graph:
		text = generate_topp(model, entry, prefix, topp, cuda_dev, max_len=max_len)
		output = output + text +  ' '
	return output