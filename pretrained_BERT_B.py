import numpy as np
import pandas as pd
import os
import sys
import zipfile
import datetime
import logging
import torch
from pytorch_pretrained_bert \
import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from collections import Counter
from run_classifier import *

# Set parameters
LabelList = ['TIN', 'UNT']
BERTModel = 'bert-base-uncased'
TrainDataPath = "Data/trainset-levelb.tsv"
TestDataPath = "Data/testset-levelb.tsv"
# ValidationRatio = 0.1
# RandomState = 1126 # Same random state gives the same validation set
NUMofEPOCH = int(sys.argv[1])
MAXSEQLEN = int(sys.argv[2])
LearningRate = 2e-5
CacheDir = "Cache"
OutputDir = "ModelB"
ResultDir = "ResultB"
# LoadModel = "ModelWeightB_01.model"
# LoadConfig = "ModelConfigB_01.config"
SaveModel = "ModelWeightB_" + str(NUMofEPOCH) + "_" + str(MAXSEQLEN) + ".model"
SaveConfig = "ModelConfigB_" + str(NUMofEPOCH) + "_" + str(MAXSEQLEN) + ".config"
SaveSubmission = "SubmissionB_" + str(NUMofEPOCH) + "_" + str(MAXSEQLEN) + ".csv"
# load_model_file = os.path.join(OutputDir, LoadModel)
# load_config_file = os.path.join(OutputDir, LoadConfig)
output_model_file = os.path.join(OutputDir, SaveModel)
output_config_file = os.path.join(OutputDir, SaveConfig)

def predict(model, tokenizer, examples, label_list, max_seq_length, eval_batch_size=128):
	device = torch.device("cuda")
	model.to(device)
	eval_examples = examples
	eval_features = convert_examples_to_features(
		eval_examples, label_list, max_seq_length, tokenizer, "classification")
	logger.info("***** Running evaluation *****")
	logger.info("  Num examples = %d", len(eval_examples))
	logger.info("  Batch size = %d", eval_batch_size)
	all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
	all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
	eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
	# Run prediction for full data
	eval_sampler = SequentialSampler(eval_data)
	eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

	model.eval()
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0
	
	res = []
	for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)
		segment_ids = segment_ids.to(device)
        # label_ids = label_ids.to(device)

		with torch.no_grad():
            # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
			logits = model(input_ids, segment_ids, input_mask)

		logits = logits.detach().cpu().numpy()
        # print(logits)
		res.extend(logits.argmax(-1))
        # label_ids = label_ids.to('cpu').numpy()
        # tmp_eval_accuracy = accuracy(logits, label_ids)

        # eval_loss += tmp_eval_loss.mean().item()
        # eval_accuracy += tmp_eval_accuracy

        # nb_eval_examples += input_ids.size(0)
		nb_eval_steps += 1

    # eval_loss = eval_loss / nb_eval_steps
    # eval_accuracy = eval_accuracy / nb_eval_examples
    # loss = tr_loss/nb_tr_steps 
    # result = {'eval_loss': eval_loss,
    #           'eval_accuracy': eval_accuracy,
    #           'global_step': global_step,
    #           'loss': loss}

    # output_eval_file = os.path.join(output_dir, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results *****")
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))
	return res

def accuracy(logits, label_ids):
	MAXID = np.argmax(logits, axis = 1)
	Count = sum(MAXID == label_ids)
	return Count

if __name__ == "__main__":
	# Load Data & cut validation & fillup nan
	Train = pd.read_csv(TrainDataPath, index_col='id', sep='\t')
	Test = pd.read_csv(TestDataPath, index_col='id', sep='\t')
	# Train, Validation = train_test_split(Train, test_size=ValidationRatio, random_state=RandomState)
	Cols = ['tweet', 'subtask']
	Train = Train.loc[:, Cols]
	Test = Test.loc[:, Cols]
	# Validation = Validation.loc[:, Cols]
	Train.fillna('UNKNOWN', inplace=True)
	Test.fillna('UNKNOWN', inplace=True)
	# Validation.fillna('UNKNOWN', inplace=True)

	# Construct input example ## guid, text_a, text_b, label
	TrainExamples = [InputExample('Train', row.tweet, label=row.subtask) for row in Train.itertuples()]
	# ValidationExamples = [InputExample('Val', row.tweet, label=row.subtask) for row in Validation.itertuples()]
	TestExamples = [InputExample('Test', row.tweet, label=LabelList[0]) for row in Test.itertuples()]

	# Set training parameters
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	n_gpu = torch.cuda.device_count()
	gradient_accumulation_steps = 1
	train_batch_size = 32
	eval_batch_size = 128
	train_batch_size = train_batch_size // gradient_accumulation_steps
	output_dir = OutputDir
	num_train_epochs = NUMofEPOCH
	num_train_optimization_steps = int(len(TrainExamples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
	cache_dir = CacheDir
	learning_rate = LearningRate
	warmup_proportion = 0.1
	max_seq_length = MAXSEQLEN

	# Load model
	tokenizer = BertTokenizer.from_pretrained(BERTModel)
	Model = BertForSequenceClassification.from_pretrained(BERTModel, cache_dir=cache_dir, num_labels=len(LabelList))
	Model.to(device)
	if n_gpu > 1:
		Model = torch.nn.DataParallel(Model)

	# Load a trained model and config that you have fine-tuned
	# tokenizer = BertTokenizer.from_pretrained(BERTModel)
	# config = BertConfig(load_config_file)
	# Model = BertForSequenceClassification(config, num_labels = len(LabelList))
	# Model.load_state_dict(torch.load(load_model_file))
	# Model.to(device)  # important to specific device
	# if n_gpu > 1:
	# 	Model = torch.nn.DataParallel(Model)

	# Prepare optimizer
	param_optimizer = list(Model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	optimizer = BertAdam(optimizer_grouped_parameters,
						 lr = learning_rate,
						 warmup = warmup_proportion,
						 t_total = num_train_optimization_steps)

	# Start training
	global_step = 0
	nb_tr_steps = 0
	tr_loss = 0

	train_features = convert_examples_to_features(
		TrainExamples, LabelList, max_seq_length, tokenizer, "classification")
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(TrainExamples))
	logger.info("  Batch size = %d", train_batch_size)
	logger.info("  Num steps = %d", num_train_optimization_steps)
	all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype = torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype = torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype = torch.long)
	all_label_ids = torch.tensor([f.label_id for f in train_features], dtype = torch.long)
	train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = train_batch_size)

	Model.train()
	for _ in trange(int(num_train_epochs), desc = "Epoch"):
		tr_loss = 0
		nb_tr_examples, nb_tr_steps = 0, 0
		total_step = len(train_data) // train_batch_size
		ten_percent_step = total_step // 10
		for step, batch in enumerate(train_dataloader):
			batch = tuple(t.to(device) for t in batch)
			input_ids, input_mask, segment_ids, label_ids = batch
			loss = Model(input_ids, segment_ids, input_mask, label_ids)
			if n_gpu > 1:
				loss = loss.mean() # mean() to average on multi-gpu.
			if gradient_accumulation_steps > 1:
				loss = loss / gradient_accumulation_steps

			loss.backward()

			tr_loss += loss.item()
			nb_tr_examples += input_ids.size(0)
			nb_tr_steps += 1
			if (step + 1) % gradient_accumulation_steps == 0:
				optimizer.step()
				optimizer.zero_grad()
				global_step += 1
				
			if step % ten_percent_step == 0:
				print("Fininshed: {:.2f}% ({}/{})".format(step/total_step*100, step, total_step))

	# Save a trained model and the associated configuration
	model_to_save = Model.module if hasattr(Model, 'module') else Model  # Only save the model it-self
	torch.save(model_to_save.state_dict(), output_model_file)
	with open(output_config_file, 'w+') as f:
		f.write(model_to_save.config.to_json_string())

	# Load a trained model and config that you have fine-tuned
	# config = BertConfig(output_config_file)
	# Model = BertForSequenceClassification(config, num_labels = len(LabelList))
	# Model.load_state_dict(torch.load(output_model_file))
	# Model.to(device)  # important to specific device
	# if n_gpu > 1:
	# 	Model = torch.nn.DataParallel(Model)\

	# Run Test
	res = predict(Model, tokenizer, TestExamples, LabelList, max_seq_length)
	cat_map = {idx:lab for idx, lab in enumerate(LabelList)}
	res = [cat_map[c] for c in res]

	#　For Submission
	Test['subtask'] = res

	submission = Test \
		.loc[:, ['subtask']] \
		.reset_index()

	submission.columns = ['id', 'subtask']
	SubmissionPath = os.path.join(ResultDir, SaveSubmission)
	submission.to_csv(SubmissionPath, index=False, header=False)
	submission.head()