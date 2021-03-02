#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import glob
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import json
import time
import logging
import random

import pandas as pd
import numpy as np
import re

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger
from pytorch_lightning.utilities import rank_zero_info

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from metrics import (
    calculate_rouge,
    calculate_bleu,
    calculate_meteor,
    calculate_chrf
    )

logger = logging.getLogger(__name__)

class WebNLG(Dataset):
    def __init__(self, tokenizer, data_dir, max_source_length,
        max_target_length, type_path,  prefix="", **dataset_kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.inputs = []
        self.targets = []
        self.dataset_kwargs = dataset_kwargs
        self._build(type_path)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        source_line = self.prefix + str(self.inputs[index]).rstrip("\n")
        tgt_line = str(self.targets[index]).rstrip("\n")
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index}

    def collate_fn(self, batch):

        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding
        
    
    def _build(self, type_path):
        if type_path == 'train':
            df = pd.read_csv('Datasets/webnlg_train.csv')
        elif type_path == 'eval':
            df = pd.read_csv('Datasets/webnlg_dev.csv')
        else:
            df = pd.read_csv('Datasets/webnlg_test.csv')
        for index, row in df.iterrows():
                line = row['input_text']
                target = row['target_text']
                self.inputs.append(line)
                self.targets.append(target)

class DART(Dataset):
    def __init__(self, tokenizer, data_dir, max_source_length,
        max_target_length, type_path,  prefix="", **dataset_kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.inputs = []
        self.targets = []
        self.dataset_kwargs = dataset_kwargs
        self._build(type_path)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        source_line = self.prefix + str(self.inputs[index]).rstrip("\n")
        tgt_line = str(self.targets[index]).rstrip("\n")
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index}

    def collate_fn(self, batch):

        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding
        
    
    def _build(self, type_path):
        if type_path == 'train':
            df = pd.read_csv('Datasets/dart_train.csv')
        elif type_path == 'eval':
            df = pd.read_csv('Datasets/dart_dev.csv')
        else:
            df = pd.read_csv('Datasets/dart_test.csv')
        for index, row in df.iterrows():
                line = row['input_text']
                target = row['target_text']
                self.inputs.append(line)
                self.targets.append(target)


class BOTH(Dataset):
    def __init__(self, tokenizer, data_dir, max_source_length,
        max_target_length, type_path,  prefix="", **dataset_kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.inputs = []
        self.targets = []
        self.dataset_kwargs = dataset_kwargs
        self._build(type_path)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        source_line = self.prefix + str(self.inputs[index]).rstrip("\n")
        tgt_line = str(self.targets[index]).rstrip("\n")
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index}

    def collate_fn(self, batch):

        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding
        
    
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

        for index, row in df.iterrows():
                line = row['input_text']
                target = row['target_text']
                self.inputs.append(line)
                self.targets.append(target)




class LoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("\n \n ***** Validation results *****")
        metrics = trainer.callback_metrics
        rank_zero_info(trainer.logger)
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("\n \n ***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))

class T5FineTuner(pl.LightningModule):

    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name_or_path)

        new_tokens = ['<H>', '<R>', '<T>']
        new_tokens_vocab = {}
        new_tokens_vocab['additional_special_tokens'] = []
        for idx, t in enumerate(new_tokens):
            new_tokens_vocab['additional_special_tokens'].append(t)
        num_added_toks = self.tokenizer.add_special_tokens(new_tokens_vocab)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.prefix = 'translate Graph to English: '

        self.eval_beams = self.hparams.eval_beams
        self.eval_max_length = self.hparams.eval_max_gen_length
        self.step_count = -2

        self.metrics = defaultdict(list)
        self.val_metric = "bleu"

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length
        }

        self.dataset_kwargs = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )

        self.num_workers = hparams.num_workers
    
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def _step(self, batch):

        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        decoder_input_ids = self.model._shift_right(tgt_ids)
    
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        return (loss,)

    @property
    def pad(self):
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx):
        loss_tensors = self._step(batch)
        logs = {name: loss for name, loss in zip(["loss"], loss_tensors)}
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        return {"loss": loss_tensors[0], "log": logs}

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def lmap(self, f, x):
        return list(map(f, x))
    
    def validation_step(self, batch, batch_idx):
        
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=None,
            num_beams=self.eval_beams,
            max_length=self.eval_max_length,
            length_penalty=1.0
        )
        
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(["loss"], loss_tensors)}
        base_metrics.update(preds=preds, target=target)
        return base_metrics
        
    def validation_epoch_end(self, outputs, prefix="val"):

        self.step_count += 1

        val_outputs_folder = "val_outputs"
        os.system("mkdir -p " + os.path.join(self.hparams.output_dir, val_outputs_folder))

        output_test_predictions_file = os.path.join(self.hparams.output_dir, val_outputs_folder, "validation_predictions_" +
                                                    str(self.step_count) + ".txt")
        output_test_targets_file = os.path.join(self.hparams.output_dir, val_outputs_folder, "validation_targets_" +
                                                    str(self.step_count) + ".txt")
        
        with open(output_test_predictions_file, "w") as p_writer, open(output_test_targets_file, "w") as t_writer:
            for output_batch in outputs:
                p_writer.writelines(convert_text(s) + "\n" for s in output_batch["preds"])
                t_writer.writelines(convert_text(s) + "\n" for s in output_batch["target"])
            p_writer.close()
            t_writer.close()

        bleu_info = calculate_bleu(output_test_targets_file, output_test_predictions_file)

        if bleu_info == -1:
            bleu_info = float(bleu_info)
        else:
            bleu_info = float(bleu_info.split(",")[0].split("BLEU = ")[1])

        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in ["loss"]}
        loss = losses["loss"]

        bleu_info: torch.FloatTensor = torch.tensor(bleu_info).type_as(loss)
        
        return {
            "bleu": bleu_info,
            f"{prefix}_loss": loss
        }

    def test_step(self, batch, batch_idx = None, dataloader_idx=None):
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=None,
            num_beams=self.eval_beams,
            max_length=self.eval_max_length,
            length_penalty=1.0
        )
        
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(["loss"], loss_tensors)}
        rouge = calculate_rouge(preds, target)
        base_metrics.update(preds=preds, target=target, **rouge)
        if dataloader_idx is not None:
            base_metrics.update(batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        return base_metrics

    def test_epoch_end(self, outputs_all_testsets, prefix="test"):

        test_outputs_folder = "test_outputs"
        os.system("mkdir -p " + os.path.join(self.hparams.output_dir, test_outputs_folder))

        output_test_predictions_file = os.path.join(self.hparams.output_dir, test_outputs_folder, "test_predictions_"
                                                    + ".txt")
        output_test_targets_file = os.path.join(self.hparams.output_dir, test_outputs_folder, "test_targets_"
                                                    + ".txt")
        
        with open(output_test_predictions_file, "w") as p_writer, open(output_test_targets_file, "w") as t_writer:
            for output_batch in outputs_all_testsets:
                p_writer.writelines(convert_text(s) + "\n" for s in output_batch["preds"])
                t_writer.writelines(convert_text(s) + "\n" for s in output_batch["target"])
            p_writer.close()
            t_writer.close()

        bleu_info = calculate_bleu(output_test_targets_file, output_test_predictions_file)
        meteor_info = calculate_meteor(output_test_targets_file, output_test_predictions_file)
        chrf_info = calculate_chrf(output_test_targets_file, output_test_predictions_file)

        if bleu_info == -1:
            bleu_info = float(bleu_info)
        else:
            bleu_info = float(bleu_info.split(",")[0].split("BLEU = ")[1])

        if meteor_info == -1:
            meteor_info = float(meteor_info)
        else:
            meteor_info = float(meteor_info.split(':')[1])

        if chrf_info == -1:
            chrf_info_doc = float(-1)
            chrf_info_avg = float(-1)
        else:
            chrf_info_doc = float(chrf_info.split(' ')[0].split("c6+w2-F2\t")[1])
            chrf_info_avg = float(chrf_info.split(' ')[1].split("c6+w2-avgF2\t")[1])

        losses = {k: torch.stack([x[k] for x in outputs_all_testsets]).mean() for k in ["loss"]}
        loss = losses["loss"]
        generative_metrics = {
            k: np.array([x[k] for x in outputs_all_testsets]).mean() for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        }

        generative_metrics['bleu'] = bleu_info

        metric_val = (
            generative_metrics['bleu'] if self.val_metric in generative_metrics else losses[
                'bleu']
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.metrics[prefix].append(all_metrics)  

        return {
            "bleu": bleu_info,
            "meteor": meteor_info,
            "chrf": chrf_info_doc,
            "chrf_avg": chrf_info_avg,
            "log": all_metrics,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    #@property
    def total_steps(self):
        num_devices = max(1, self.hparams.gpus) 
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        dataset_size = len(self.train_loader.dataset)

        total_steps = (dataset_size / effective_batch_size) * self.hparams.max_epochs
        return total_steps

    def setup(self,mode):
        self.train_loader = self.train_dataloader()

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.hparams.weight_decay,
                },
                {
                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer

        scheduler = get_linear_schedule_with_warmup(
                self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def get_dataset(self, type_path) -> DART:
        max_target_length = self.target_lens[type_path]
        dataset = DART(
            self.tokenizer,
            type_path=type_path,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset
    
    
    def train_dataloader(self):
        dataset = self.get_dataset("train")
        batch_size=self.hparams.train_batch_size
        return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=True,
                num_workers=self.num_workers,
                sampler=None
                )

    def val_dataloader(self):
        dataset = self.get_dataset("val")
        batch_size=self.hparams.eval_batch_size
        return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=None
                )

    def test_dataloader(self):
        dataset = self.get_dataset("test")
        batch_size=self.hparams.eval_batch_size
        return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=None
                )

    def get_progress_bar_dict(self):
        lrs = self.trainer.lr_logger.lrs['lr-AdamW/pg1'][-1]
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        tqdm_dict = {"loss": "{:.3f}".format(avg_training_loss), "lr": lrs}
        return tqdm_dict
    
    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        
        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
            required=True,
            help="The output directory where the model predictions and checkpoints will be written.",)
        
        parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
        parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )

        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,  
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument("--checkpoint", type=str, default=None, required=False)
        parser.add_argument(
            "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )

        return parser


def flatten_list(y):
    return [x for x in itertools.chain.from_iterable(y)]

def convert_text(text):
    text = text.lower()
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text

def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False):
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{step_count}"
    elif metric == "loss":
        exp = "{val_avg_loss:.4f}-{step_count}"
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function."
        )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(output_dir, exp),
        monitor=metric,
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
        period=0,
    )
    return checkpoint_callback

def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=metric,
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )

def main(args):

    pl.seed_everything(args.seed)

    Path(args.output_dir).mkdir(exist_ok=True)
    model = T5FineTuner(args)
    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    dataset = Path(args.data_dir).name
    logger = True
    
    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback("bleu", args.early_stopping_patience)
    else:
        es_callback = False

    val_metric_loss = args.val_metric == "loss"

    checkpoint_callback=get_checkpoint_callback(
            args.output_dir, "bleu", args.save_top_k, val_metric_loss
        )
    
    train_params = {}
    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches

    lr_logger = LearningRateLogger(logging_interval='step')

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks = [LoggingCallback(),lr_logger],
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=es_callback,
        num_sanity_val_steps=4,
        **train_params
        )
    
    trainer.lr_logger = lr_logger

    if args.do_train:
        trainer.fit(model)

    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    
    if not args.checkpoint:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    else:
        checkpoints = [args.checkpoint]

    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]

        if args.do_predict and not args.do_train:

            checkpoint = checkpoints[-1]
            print(checkpoint)
            trainer.test(model, ckpt_path=checkpoint)
            return model

    trainer.test()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = T5FineTuner.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    main(args)