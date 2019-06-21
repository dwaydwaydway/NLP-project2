import numpy as np
import pandas as pd
import os
import sys
import zipfile
import datetime
import logging
import torch
from pytorch_pretrained_bert \
    import BertTokenizer, BertModel, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from collections import Counter
from run_classifier import *
import torch
from torch import nn
from torch.nn.functional import cross_entropy, softmax
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
import warnings
warnings.filterwarnings("ignore")
import math
from loss import FocalLoss
# Set parameters
LabelList = ['IND', 'GRP', 'OTH']
BERTModel = 'bert-base-uncased'
TrainDataPath = "Data/trainset-levelc.tsv"
TestDataPath = "Data/testset-levelc.tsv"
ValidationRatio = 0.1
RandomState = 1126  # Same random state gives the same validation set
NUMofEPOCH = int(sys.argv[1])
MAXSEQLEN = int(sys.argv[2])
BertLR = 2e-5
classifierLR = 1e-3
CacheDir = "Cache"
OutputDir = "ModelC"
ResultDir = "ResultC"
# LoadModel = "ModelWeightC_01.model"
# LoadConfig = "ModelConfigC_01.config"
SaveModel = "f1_ModelWeightC_" + \
    str(NUMofEPOCH) + "_" + str(MAXSEQLEN) + ".model"
SaveConfig = "f1_ModelConfigC_" + \
    str(NUMofEPOCH) + "_" + str(MAXSEQLEN) + ".config"
SaveSubmission = "f1_SubmissionC_" + \
    str(NUMofEPOCH) + "_" + str(MAXSEQLEN) + ".csv"
# load_model_file = os.path.join(OutputDir, LoadModel)
# load_config_file = os.path.join(OutputDir, LoadConfig)
output_model_file = os.path.join(OutputDir, SaveModel)
output_config_file = os.path.join(OutputDir, SaveConfig)


def custom_loss(predict, labels):
    CE = cross_entropy(predict, labels)
    y_onehot = torch.FloatTensor(labels.size(0), predict.size(1)).cuda()
    y_onehot.zero_()
    target = y_onehot.scatter_(1, labels.unsqueeze(1), 1)
    predict = softmax(predict, dim=1)
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return (1 - f1.mean()), CE


class F1_BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(F1_BertForSequenceClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=output_attentions,
                              keep_multihead_output=keep_multihead_output)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        self.focal_loss = FocalLoss(num_labels)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask,
                            output_all_encoded_layers=False, head_mask=head_mask)
        if self.output_attentions:
            all_attentions, _, pooled_output = outputs
        else:
            _, pooled_output = outputs
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            return custom_loss(logits.view(-1, self.num_labels), labels.view(-1))[0], self.focal_loss(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.output_attentions:
            return all_attentions, logits
        return logits


def predict(model, tokenizer, examples, label_list, max_seq_length, eval_batch_size=128):
    device = torch.device("cuda")
    model.to(device)
    eval_examples = examples
    eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer, "classification")
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

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
        nb_eval_steps += 1
    return res


def accuracy(logits, label_ids):
    MAXID = np.argmax(logits, axis=1)
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
    TrainExamples = [InputExample(
        'Train', row.tweet, label=row.subtask) for row in Train.itertuples()]
    # ValidationExamples = [InputExample('Val', row.tweet, label=row.subtask) for row in Validation.itertuples()]
    TestExamples = [InputExample(
        'Test', row.tweet, label=LabelList[0]) for row in Test.itertuples()]

    # Set training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    gradient_accumulation_steps = 1
    train_batch_size = 32
    eval_batch_size = 128
    train_batch_size = train_batch_size // gradient_accumulation_steps
    output_dir = OutputDir
    num_train_epochs = NUMofEPOCH
    num_train_optimization_steps = math.ceil(len(
        TrainExamples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
    cache_dir = CacheDir
    warmup_proportion = 0.1
    max_seq_length = MAXSEQLEN

    # Load model
    tokenizer = BertTokenizer.from_pretrained(BERTModel)
    Model = F1_BertForSequenceClassification.from_pretrained(
        BERTModel, cache_dir=cache_dir, num_labels=len(LabelList))
    Model.to(device)
    if n_gpu > 1:
        Model = torch.nn.DataParallel(Model)

    # Prepare optimizer
    param_optimizer = list(Model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': Model.bert.parameters(), 'weight_decay': 0.01, 'lr': BertLR},
        {'params': Model.classifier.parameters(), 'weight_decay': 0.0,
         'lr': classifierLR}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         warmup=warmup_proportion,
                         t_total=num_train_optimization_steps)

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
    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=train_batch_size)

    Model.train()
    for epoch in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        total_step = len(train_data) // train_batch_size
        ten_percent_step = total_step // 10
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            f1, ce = Model(input_ids, segment_ids, input_mask, label_ids)
            alpha = epoch / float(num_train_epochs)
            loss = 0.2 * f1 + 0.8 * ce
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
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

    # Save a trained model and the associated configuration
    model_to_save = Model.module if hasattr(
        Model, 'module') else Model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)
    with open(output_config_file, 'w+') as f:
        f.write(model_to_save.config.to_json_string())

    # Run Test
    res = predict(Model, tokenizer, TestExamples, LabelList, max_seq_length)
    cat_map = {idx: lab for idx, lab in enumerate(LabelList)}
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
