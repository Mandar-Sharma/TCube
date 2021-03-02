#!/usr/bin/env python
# coding: utf-8

import os

from rouge_score import rouge_scorer, scoring

def calculate_meteor(target_file, pred_file):

	cmd_string = "java -Xmx2G -jar WordBasedMetrics/meteor-1.5.jar " + pred_file + " " + target_file + " -l en -norm -r 1 > " + pred_file.replace("txt", "meteor")
	os.system(cmd_string)
	try:
		meteor_info = open(pred_file.replace("txt", "meteor"), 'r').readlines()[-1].strip()
	except:
		meteor_info = -1		

	return meteor_info

def calculate_rouge(pred_lns,tgt_lns,use_stemmer=True, bootstrap_aggregation=True,newline_sep=True):
	
	scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=use_stemmer)
	aggregator = scoring.BootstrapAggregator()
	for pred, tgt in zip(tgt_lns, pred_lns):
		scores = scorer.score(pred, tgt)
		aggregator.add_scores(scores)

	if bootstrap_aggregation:
		result = aggregator.aggregate()
		return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

	else:
		return aggregator._scores

def calculate_bleu(target_file, pred_file):

	cmd_string = "perl WordBasedMetrics/multi-bleu.perl -lc " + target_file + " < " + pred_file + " > " + pred_file.replace("txt", "bleu")
	os.system(cmd_string)
	try:
		bleu_info = open(pred_file.replace("txt", "bleu"), 'r').readlines()[0].strip()
	except:
		bleu_info = -1

	return bleu_info

def calculate_chrf(target_file, pred_file):

	cmd_string = "python WordBasedMetrics/chrf++.py -H " + pred_file + " -R " \
				  + target_file + " > " + pred_file.replace("txt", "chrf")
	os.system(cmd_string)
	try:
		chrf_info_1 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[1].strip()
		chrf_info_2 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[2].strip()
		chrf_info = chrf_info_1 + " " + chrf_info_2
	except:
		chrf_info = -1

	return chrf_info