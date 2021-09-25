# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

#from emmental import MaskedBertConfig, MaskedBertForSequenceClassification

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  BertForSequenceClassification,
                                  AlbertConfig,
                                  AlbertTokenizer,
                                  RobertaConfig,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)

from transformers.modeling_highway_bert import BertForSequenceClassification as BertForSequenceClassificationHW
from transformers.modeling_highway_albert import AlbertForSequenceClassification as AlbertForSequenceClassificationHW

from transformers.modeling_albert import AlbertForSequenceClassification as AlbertForSequenceClassification

#need to add imports to use bert model
from transformers.modeling_bert_masked import MaskedBertConfig #MaskedBertForSequenceClassification
from transformers.modeling_albert_masked import MaskedAlbertConfig #MaskedAlbertForSequenceClassification
from transformers.modeling_highway_bert_masked import MaskedBertForSequenceClassification
from transformers.modeling_highway_albert_masked import MaskedAlbertForSequenceClassification

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (AlbertConfig, BertConfig, XLNetConfig, XLMConfig,
                                                                                RobertaConfig, DistilBertConfig)), ())

#'bert': (BertConfig, MaskedBertForSequenceClassification, BertTokenizer),
#'albert': (AlbertConfig, MaskedAlbertForSequenceClassification, AlbertTokenizer),
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassificationHW, BertTokenizer),
    'bert_teacher': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassificationHW, AlbertTokenizer),
    'albert_teacher': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'masked_bert': (MaskedBertConfig, MaskedBertForSequenceClassification, BertTokenizer),
    'masked_albert': (MaskedAlbertConfig, MaskedAlbertForSequenceClassification, AlbertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_wanted_result(result):
    if "spearmanr" in result:
        print_result = result["spearmanr"]
    elif "f1" in result:
        print_result = result["f1"]
    elif "mcc" in result:
        print_result = result["mcc"]
    elif "acc" in result:
        print_result = result["acc"]
    else:
        print(result)
        exit(1)
    return print_result

def schedule_threshold(
    step: int,
    total_step: int,
    warmup_steps: int,
    initial_threshold: float,
    final_threshold: float,
    initial_warmup: int,
    final_warmup: int,
    final_lambda: float,
):
    if step <= initial_warmup * warmup_steps:
        threshold = initial_threshold
    elif step > (total_step - final_warmup * warmup_steps):
        threshold = final_threshold
    else:
        spars_warmup_steps = initial_warmup * warmup_steps
        spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
    regu_lambda = final_lambda * threshold / final_threshold
    return threshold, regu_lambda

def regularization(model: nn.Module, mode: str):
    regu, counter = 0, 0
    for name, param in model.named_parameters():
        if "mask_scores" in name:
            if mode == "l1":
                regu += torch.norm(torch.sigmoid(param), p=1) / param.numel()
            elif mode == "l0":
                regu += torch.sigmoid(param - 2 / 3 * np.log(0.1 / 1.1)).sum() / param.numel()
            else:
                ValueError("Don't know this mode.")
            counter += 1
    return regu / counter

#def train(args, train_dataset, model, tokenizer, prune_schedule=None, train_highway=False):
def train(args, train_dataset, model, tokenizer, teacher=None, prune_schedule=None, train_highway=False):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    if train_highway:
        optimizer_grouped_parameters = [
            {
                 "params": [p for n, p in model.named_parameters() if ("highway" in n) and "mask_score" in n and p.requires_grad],
                 "lr": args.mask_scores_learning_rate,
            },
            {
                 "params": [
                     p
                     for n, p in model.named_parameters()
                     if "mask_score" not in n and p.requires_grad and ("highway" in n) and not any(nd in n for nd in no_decay)
                 ],
                 "lr": args.learning_rate,
                 "weight_decay": args.weight_decay,
            },
            {
                 "params": [
                     p
                     for n, p in model.named_parameters()
                     if "mask_score" not in n and p.requires_grad and ("highway" in n) and any(nd in n for nd in no_decay)
                 ],
                 "lr": args.learning_rate,
                 "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                 "params": [p for n, p in model.named_parameters() if ("highway" not in n) and "mask_score" in n and p.requires_grad],
                 "lr": args.mask_scores_learning_rate,
            },
            {
                 "params": [
                     p
                     for n, p in model.named_parameters()
                     if ("highway" not in n) and "mask_score" not in n and p.requires_grad and not any(nd in n for nd in no_decay)
                 ],
                 "lr": args.learning_rate,
                 "weight_decay": args.weight_decay,
            },
            {
                 "params": [
                     p
                     for n, p in model.named_parameters()
                     if ("highway" not in n) and "mask_score" not in n and p.requires_grad and any(nd in n for nd in no_decay)
                 ],
                 "lr": args.learning_rate,
                 "weight_decay": 0.0,
            },
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    if teacher is not None:
        logger.info("  Training with distillation")

    global_step = 0
    # Global TopK
    if args.global_topk:
        threshold_mem = None

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    epoch = 1

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        if args.fxp_and_prune:
           if epoch >= args.start_epoch:
               prune_amount = prune_schedule[epoch - args.start_epoch]
               print ("prune amount is: ", prune_amount)

        for step, batch in enumerate(epoch_iterator):
#             if steps_trained_in_current_epoch > 0:
#                 steps_trained_in_current_epoch -= 1
#                 continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            threshold, regu_lambda = schedule_threshold(
                step=global_step,
                total_step=t_total,
                warmup_steps=args.warmup_steps,
                final_threshold=args.final_threshold,
                initial_threshold=args.initial_threshold,
                final_warmup=args.final_warmup,
                initial_warmup=args.initial_warmup,
                final_lambda=args.final_lambda,
            )

            # Global TopK
            if args.global_topk:
                if threshold == 1.0:
                    threshold = -1e2  # Or an indefinitely low quantity
                else:
                    if (threshold_mem is None) or (global_step % args.global_topk_frequency_compute == 0):
                        # Sort all the values to get the global topK
                        concat = torch.cat(
                            [param.view(-1) for name, param in model.named_parameters() if "mask_scores" in name]
                        )
                        n = concat.numel()
                        kth = max(n - (int(n * threshold) + 1), 1)
                        threshold_mem = concat.kthvalue(kth).values.item()
                        threshold = threshold_mem
                    else:
                        threshold = threshold_mem

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert': # and args.model_type != 'albert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'masked_bert', 'bert_teacher', 'albert', 'masked_albert', 'albert_teacher'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            inputs['train_highway'] = train_highway
            if "masked" in args.model_type:
                 inputs["threshold"] = threshold

            outputs = model(**inputs)
            #loss, logits_stu = outputs  # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            logits_stu = outputs[1]
            # Distillation loss
            if teacher is not None:
                if "token_type_ids" not in inputs:
                    inputs["token_type_ids"] = None if args.teacher_type == "xlm" else batch[2]
                with torch.no_grad():
                    (logits_tea,) = teacher(
                        input_ids=inputs["input_ids"],
                        token_type_ids=inputs["token_type_ids"],
                        attention_mask=inputs["attention_mask"],
                    )

                loss_logits = F.kl_div(
                    input=F.log_softmax(logits_stu / args.temperature, dim=-1),
                    target=F.softmax(logits_tea / args.temperature, dim=-1),
                    reduction="batchmean",
                ) * (args.temperature ** 2)

                loss = args.alpha_distil * loss_logits + args.alpha_ce * loss


            # Regularization
            if args.regularization is not None:
                regu_ = regularization(model=model, mode=args.regularization)
                loss = loss + regu_lambda * regu_

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Adaptive Attention Span Loss
            if args.adaptive:
               adapt_span_loss = 0.0
               for l in model.albert.encoder.albert_layer_groups:
                   for ll in l.albert_layers:
                       adapt_span_loss += ll.attention.adaptive_span.get_loss()
               loss += adapt_span_loss

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            #if (step + 1) % args.gradient_accumulation_steps == 0:
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.adaptive:
               for l in model.albert.encoder.albert_layer_groups:
                   for ll in l.albert_layers:
                       ll.attention.adaptive_span.clamp_param()

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.fxp_and_prune:
            if epoch >= args.start_epoch:
                for name, parameter in model.named_parameters():
                    if "embedding" in name:
                    #if name in args.prune_names:
                        weights        = parameter.abs().cpu().data.numpy()
                        threshold      = np.percentile(weights, prune_amount)
                        mask           = (torch.zeros(parameter.size()) + threshold).cuda()
                        mask           = parameter.abs().ge(Variable(mask)).float()
                        parameter.data = parameter.data * mask.data

        if args.adaptive:
           logger.info("Adaptive Span Loss: %s", str(adapt_span_loss.item()))

        if args.adaptive:
           for layer_idx1, i in enumerate(model.albert.encoder.albert_layer_groups):
               for layer_idx2, j in enumerate (i.albert_layers):
                   k = j.attention.adaptive_span.get_current_avg_span()
                   ms = j.attention.adaptive_span.get_current_max_span()
                   logger.info("Avg Attn Span for layer %d,%d =%d\t", layer_idx1, layer_idx2, k)
                   logger.info("Max Attn Span for layer %d,%d =%d\t", layer_idx1, layer_idx2, ms)

        epoch += 1

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", output_layer=-1, eval_highway=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        exit_layer_counter = {(i+1):0 for i in range(model.num_layers)}
        st = time.time()

        # Global TopK
        if args.global_topk:
            threshold_mem = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert': # and args.model_type != 'albert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'masked_bert', 'bert_teacher', 'albert', 'masked_albert', 'albert_teacher'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                if "masked" in args.model_type:
                    inputs["threshold"] = args.final_threshold
                    if args.global_topk:
                        if threshold_mem is None:
                            concat = torch.cat(
                                [param.view(-1) for name, param in model.named_parameters() if "mask_scores" in name]
                            )
                            n = concat.numel()
                            kth = max(n - (int(n * args.final_threshold) + 1), 1)
                            threshold_mem = concat.kthvalue(kth).values.item()
                        inputs["threshold"] = threshold_mem

                if output_layer >= 0:
                    inputs['output_layer'] = output_layer

                outputs = model(**inputs)
                if eval_highway:
                    exit_layer_counter[outputs[-1]] += 1
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_time = time.time() - st
        print("Eval time:", eval_time)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        if eval_highway:
            print("Exit layer counter", exit_layer_counter)
            num_ex = sum(exit_layer_counter.values())
            running_avg = 0
            for i in range(model.num_layers):
                running_avg += exit_layer_counter[i+1] * (i+1)
            avg_exit_layer = running_avg / num_ex
            if (args.get_predict_acc): #getting diff between EE and predict
                print("Average Layer Diff", avg_exit_layer-1)
            else:
                print("Average Exit Layer", avg_exit_layer)
            actual_cost = sum([l*c for l, c in exit_layer_counter.items()])
            full_cost = len(eval_dataloader) * model.num_layers
            print("Expected saving", actual_cost/full_cost)
            if args.early_exit_entropy>=0:
                save_fname = args.plot_data_dir + '/' +\
                             args.model_name_or_path[2:] +\
                             "/entropy_{}.npy".format(args.early_exit_entropy)
                if not os.path.exists(os.path.dirname(save_fname)):
                    os.makedirs(os.path.dirname(save_fname))
                print_result = get_wanted_result(result)
                print(print_result)
                np.save(save_fname,
                        np.array([exit_layer_counter,
                                  eval_time,
                                  actual_cost/full_cost,
                                  print_result]))

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--plot_data_dir", default="./plotting/", type=str, required=False,
                        help="The directory to store data for plotting figures.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_each_highway", action='store_true',
                        help="Set this flag to evaluate each highway.")
    parser.add_argument("--eval_after_first_stage", action='store_true',
                        help="Set this flag to evaluate after training only bert (not highway).")
    parser.add_argument("--eval_highway", action='store_true',
                        help="Set this flag if it's evaluating highway models")
    parser.add_argument(
        "--one_class",
        action='store_true',
        help="Set this flag to use only one highway classifier",
    )
    parser.add_argument(
        "--entropy_predictor",
        action='store_true',
        help="Set this flag to do entropy prediction",
    )
    parser.add_argument("--predict_layer", default=1, type=int,
                        help="Layer to perform entropy prediction")
    parser.add_argument("--predict_average_layers", default=0, type=int,
                        help="Whether to average entropy values when doing entropy prediction.")
    parser.add_argument('--lookup_table_file', type=str, default='./sst2_lookup_table.csv',
                        help="Path to lookup table")
    parser.add_argument(
        "--extra_layer",
        action='store_true',
        help="Set this flag to allow for an extra layer after the predicted layer",
    )
    parser.add_argument(
        "--get_predict_acc",
        action='store_true',
        help="Set this flag to compare prediction acc w/ EE acc",
    )
    parser.add_argument(
        "--no_ee_before",
        action='store_true',
        help="Set this flag to not perform ee before the predicted layer",
    )


    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--early_exit_entropy", default=-1, type=float,
                        help = "Entropy threshold for early exit.")


    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Adaptive Attention Span
    parser.add_argument('--adaptive', action='store_true', help="Enable Adaptive Attention Span")
    parser.add_argument('--adaptive_span_ramp', type=int, default=256, help="Adaptive Attention Span Ramp")
    parser.add_argument('--max_span', type=int, default=512, help="Adaptive Attention Span Ramp")

    #magnitude pruning (use for embeddings)
    parser.add_argument('--fxp_and_prune', action='store_true', help="For mag pruning of embeddings")
    parser.add_argument('--start_epoch', type=int, default=1, help="For pruning.")
    parser.add_argument('--prune_percentile', type=int, default=0, help="For pruning.")
    parser.add_argument('--prune_names', type=str, default='', help="For pruning.")

    # Pruning parameters
    parser.add_argument(
        "--mask_scores_learning_rate",
        default=1e-2,
        type=float,
        help="The Adam initial learning rate of the mask scores.",
    )
    parser.add_argument(
        "--initial_threshold", default=1.0, type=float, help="Initial value of the threshold (for scheduling)."
    )
    parser.add_argument(
        "--final_threshold", default=0.7, type=float, help="Final value of the threshold (for scheduling)."
    )
    parser.add_argument(
        "--initial_warmup",
        default=1,
        type=int,
        help="Run `initial_warmup` * `warmup_steps` steps of threshold warmup during which threshold stays"
        "at its `initial_threshold` value (sparsity schedule).",
    )
    parser.add_argument(
        "--final_warmup",
        default=2,
        type=int,
        help="Run `final_warmup` * `warmup_steps` steps of threshold cool-down during which threshold stays"
        "at its final_threshold value (sparsity schedule).",
    )

    parser.add_argument(
        "--pruning_method",
        default="topK",
        type=str,
        help="Pruning Method (l0 = L0 regularization, magnitude = Magnitude pruning, topK = Movement pruning, sigmoied_threshold = Soft movement pruning).",
    )
    parser.add_argument(
        "--mask_init",
        default="constant",
        type=str,
        help="Initialization method for the mask scores. Choices: constant, uniform, kaiming.",
    )
    parser.add_argument(
        "--mask_scale", default=0.0, type=float, help="Initialization parameter for the chosen initialization method."
    )

    parser.add_argument("--regularization", default=None, help="Add L0 or L1 regularization to the mask scores.")
    parser.add_argument(
        "--final_lambda",
        default=0.0,
        type=float,
        help="Regularization intensity (used in conjunction with `regulariation`.",
    )

    parser.add_argument("--global_topk", action="store_true", help="Global TopK on the Scores.")
    parser.add_argument(
        "--global_topk_frequency_compute",
        default=25,
        type=int,
        help="Frequency at which we compute the TopK global threshold.",
    )

    # Distillation parameters (optional)
    parser.add_argument(
        "--teacher_type",
        default=None,
        type=str,
        help="Teacher type. Teacher tokenizer and student (model) tokenizer must output the same tokenization. Only for distillation.",
    )
    parser.add_argument(
        "--teacher_name_or_path",
        default=None,
        type=str,
        help="Path to the already fine-tuned teacher model. Only for distillation.",
    )
    parser.add_argument(
        "--alpha_ce", default=0.5, type=float, help="Cross entropy loss linear weight. Only for distillation."
    )
    parser.add_argument(
        "--alpha_distil", default=0.5, type=float, help="Distillation loss linear weight. Only for distillation."
    )
    parser.add_argument(
        "--temperature", default=2.0, type=float, help="Distillation temperature. Only for distillation."
    )

    args = parser.parse_args()

    #Adaptive Attention Span Params
    params = {
      "adapt_span_enabled": args.adaptive,
      "attn_span": args.max_span,
      "adapt_span_loss_coeff": 0.000005,
      "adapt_span_ramp": args.adaptive_span_ramp,
      "adapt_span_init": 0.002,
      "adapt_span_cache": True,
      "nb_heads": 12,
      "bs": args.per_gpu_train_batch_size,
      "mask_size": [0, 128],
    }

    # Regularization
    if args.regularization == "null":
        args.regularization = None

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
    #                                       num_labels=num_labels,
    #                                       finetuning_task=args.task_name,
    #                                       cache_dir=args.cache_dir if args.cache_dir else None)
    # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #                                             do_lower_case=args.do_lower_case,
    #                                             cache_dir=args.cache_dir if args.cache_dir else None)
    #
    # #sys.stdout.flush()
    # model = model_class.from_pretrained(args.model_name_or_path,
    #                                     from_tf=bool('.ckpt' in args.model_name_or_path),
    #                                     config=config,
    #                                     cache_dir=args.cache_dir if args.cache_dir else None)
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
        pruning_method=args.pruning_method,
        mask_init=args.mask_init,
        mask_scale=args.mask_scale,
        one_class=args.one_class,
        entropy_predictor=args.entropy_predictor,
        predict_layer=args.predict_layer,
        predict_average_layers=args.predict_average_layers,
        lookup_table_file=args.lookup_table_file,
        extra_layer=args.extra_layer,
        get_predict_acc=args.get_predict_acc,
        no_ee_before=args.no_ee_before
    )

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        do_lower_case=args.do_lower_case,
    )
    #print ("check2\n", flush=True)
    #print ("printing params: ", params, flush=True)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        params=params,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.teacher_type is not None:
        assert args.teacher_name_or_path is not None
        assert args.alpha_distil > 0.0
        assert args.alpha_distil + args.alpha_ce > 0.0
        teacher_config_class, teacher_model_class, _ = MODEL_CLASSES[args.teacher_type]
        teacher_config = teacher_config_class.from_pretrained(args.teacher_name_or_path)
        teacher = teacher_model_class.from_pretrained(
            args.teacher_name_or_path,
            from_tf=False,
            config=teacher_config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        teacher.to(args.device)
    else:
        teacher = None

    if args.model_type == "bert" or args.model_type == "masked_bert":
        model.bert.encoder.set_early_exit_entropy(args.early_exit_entropy)
        model.bert.init_highway_pooler()
    elif args.model_type == "albert" or args.model_type == "masked_albert":
        model.albert.encoder.set_early_exit_entropy(args.early_exit_entropy)
        model.albert.init_highway_pooler()

    if args.fxp_and_prune:
         n_train_epochs = int(args.num_train_epochs)
         prune_schedule = [ 0 for _ in range(n_train_epochs + 1 - args.start_epoch) ]

         start_prune_epoch = int(0.1 * (n_train_epochs + 1 - args.start_epoch))
         end_prune_epoch = int(0.8 * (n_train_epochs + 1 - args.start_epoch))

         ramp_epochs = end_prune_epoch - start_prune_epoch
         for i in range(start_prune_epoch , end_prune_epoch ):
           prune_schedule[i] = float((i - start_prune_epoch + 1)) / ramp_epochs * (args.prune_percentile - prune_schedule[0]) + prune_schedule[0]
         for i in range(end_prune_epoch, (n_train_epochs + 1 - args.start_epoch)):
           prune_schedule[i] = args.prune_percentile

         print("Prune schedule is: ", prune_schedule)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        if args.fxp_and_prune:
            global_step, tr_loss = train(args, train_dataset, model, tokenizer, teacher=teacher, prune_schedule=prune_schedule)
        else:
            global_step, tr_loss = train(args, train_dataset, model, tokenizer, teacher=teacher)
        # global_step, tr_loss = train(args, train_dataset, model, tokenizer, teacher=teacher)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if args.eval_after_first_stage:
            result = evaluate(args, model, tokenizer, prefix="")
            print_result = get_wanted_result(result)

        if args.fxp_and_prune:
            train(args, train_dataset, model, tokenizer, teacher=teacher, prune_schedule=prune_schedule, train_highway=True)
        else:
            train(args, train_dataset, model, tokenizer, teacher=teacher, train_highway=True)
        # train(args, train_dataset, model, tokenizer, teacher=teacher, train_highway=True)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, params=params)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = model_class.from_pretrained(checkpoint, params=params, config=config)
            if args.model_type=="bert":
                model.bert.encoder.set_early_exit_entropy(args.early_exit_entropy)
            elif args.model_type=="masked_bert":
                model.bert.encoder.set_early_exit_entropy(args.early_exit_entropy)
            elif args.model_type=="albert":
                model.albert.encoder.set_early_exit_entropy(args.early_exit_entropy)
            elif args.model_type=="masked_albert":
                model.albert.encoder.set_early_exit_entropy(args.early_exit_entropy)
            #else:
            #    model.roberta.encoder.set_early_exit_entropy(args.early_exit_entropy)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix,
                              eval_highway=args.eval_highway)
            print_result = get_wanted_result(result)
            print("Result: {}".format(print_result))
            if args.eval_each_highway:
                last_layer_results = print_result
                each_layer_results = []
                for i in range(model.num_layers):
                    logger.info("\n")
                    _result = evaluate(args, model, tokenizer, prefix=prefix,
                                       output_layer=i, eval_highway=args.eval_highway)
                    if i+1 < model.num_layers:
                        each_layer_results.append(get_wanted_result(_result))
                each_layer_results.append(last_layer_results)
                save_fname = args.plot_data_dir + '/' + args.model_name_or_path[2:] + "/each_layer.npy"
                if not os.path.exists(os.path.dirname(save_fname)):
                    os.makedirs(os.path.dirname(save_fname))
                np.save(save_fname,
                        np.array(each_layer_results))
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
