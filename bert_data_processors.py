import os
import csv
import collections
import sys
if not 'texar_repo' in sys.path:
  print("oh no")
  sys.path += ['texar_repo']
  print(sys.path)
from texar_repo.examples.bert.utils import data_utils, model_utils, tokenization
import importlib
import tensorflow as tf
import texar as tx
from texar_repo.examples.bert import config_classifier as config_downstream
from texar_repo.texar.utils import transformer_utils
from texar_repo.examples.transformer.utils import data_utils, utils
from texar_repo.examples.transformer.bleu_tool import bleu_wrapper
from flask import Flask,request,render_template
import requests
import json
from collections import OrderedDict
import os
import numpy as np
import json
#import rouge
from rouge import Rouge

















class InputExample():

    def __init__(self, guid, text_a, text_b=None):
        self.guid = guid
        self.src_txt = text_a
        self.tgt_txt = text_b
        
class InputFeatures():

    def __init__(self, src_input_ids,src_input_mask,src_segment_ids,tgt_input_ids,tgt_input_mask,tgt_labels):
        self.src_input_ids = src_input_ids
        self.src_input_mask = src_input_mask
        self.src_segment_ids = src_segment_ids
        self.tgt_input_ids = tgt_input_ids
        self.tgt_input_mask = tgt_input_mask 
        self.tgt_labels = tgt_labels
        
       
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                lines.append(line)
        return lines


    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\n", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                lines.append(line)
        return lines
      
      
class CNNDailymail(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "/var/scratch/vro220/train_story_combine.txt")),self._read_file(os.path.join(data_dir, "/var/scratch/vro220/train_summ_combine.txt")),
            "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "/var/scratch/vro220/eval_story_combine.txt")),self._read_file(os.path.join(data_dir, "/var/scratch/vro220/eval_summ_combine.txt")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "/var/scratch/vro220/test_story_combine.txt")),self._read_file(os.path.join(data_dir, "/var/scratch/vro220/test_summ_combine.txt")),
            "test")

    def _create_examples(self, src_lines,tgt_lines,set_type):
        examples = [] 
        for i,data in enumerate(zip(src_lines,tgt_lines)):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test" and i == 0:
                continue
            else:
                #print(data)
                if len(data[0])==0 or len(data[1])==0:
                  continue
                src_lines = tokenization.convert_to_unicode(data[0][0])
                tgt_lines = tokenization.convert_to_unicode(data[1][0])
                examples.append(InputExample(guid=guid, text_a=src_lines,
                                         text_b=tgt_lines))
        return examples
  
  
def file_based_convert_examples_to_features(
        examples, max_seq_length_src,max_seq_length_tgt,tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        #print("ex_index",ex_index)

        if (ex_index+1) %1000 == 0 :
          print("------------processed..{}...examples".format(ex_index))
          
        feature = convert_single_example(ex_index, example,
                                         max_seq_length_src,max_seq_length_tgt,tokenizer)

        def create_int_feature(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["src_input_ids"] = create_int_feature(feature.src_input_ids)
        features["src_input_mask"] = create_int_feature(feature.src_input_mask)
        features["src_segment_ids"] = create_int_feature(feature.src_segment_ids)

        features["tgt_input_ids"] = create_int_feature(feature.tgt_input_ids)
        features["tgt_input_mask"] = create_int_feature(feature.tgt_input_mask)
        features['tgt_labels'] = create_int_feature(feature.tgt_labels)
        
        
        
        #print(feature.tgt_labels)
        

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def convert_single_example(ex_index, example, max_seq_length_src,max_seq_length_tgt,
                           tokenizer):
    
    tokens_a = tokenizer.tokenize(example.src_txt)
    tokens_b = tokenizer.tokenize(example.tgt_txt)
    if len(tokens_a) > max_seq_length_src - 2:
            tokens_a = tokens_a[0:(max_seq_length_src - 2)]
    
    if len(tokens_b) > max_seq_length_tgt - 2:
            tokens_b = tokens_b[0:(max_seq_length_tgt - 2)]

    
    tokens_src = []
    segment_ids_src = []
    tokens_src.append("[CLS]")
    segment_ids_src.append(0)
    for token in tokens_a:
        tokens_src.append(token)
        segment_ids_src.append(0)
    tokens_src.append("[SEP]")
    segment_ids_src.append(0)
  

    tokens_tgt = []
    segment_ids_tgt = []
    tokens_tgt.append("[CLS]")
    #segment_ids_tgt.append(0)
    for token in tokens_b:
        tokens_tgt.append(token)
        #segment_ids_tgt.append(0)
    tokens_tgt.append("[SEP]")
    #segment_ids_tgt.append(0)

    input_ids_src = tokenizer.convert_tokens_to_ids(tokens_src)
   
    

    input_ids_tgt = tokenizer.convert_tokens_to_ids(tokens_tgt)
    
    #Adding begiining and end token
    labels_tgt = input_ids_tgt[1:]
    input_ids_tgt = input_ids_tgt[:-1] 
    input_mask_src = [1] * len(input_ids_src)
    input_mask_tgt = [1] * len(input_ids_tgt)
    
    
    
    while len(input_ids_src) < max_seq_length_src:
        input_ids_src.append(0)
        input_mask_src.append(0)
        segment_ids_src.append(0)

    while len(input_ids_tgt) < max_seq_length_tgt:
        input_ids_tgt.append(0)
        input_mask_tgt.append(0)
        segment_ids_tgt.append(0)
        labels_tgt.append(0)

    feature = InputFeatures( src_input_ids=input_ids_src,src_input_mask=input_mask_src,src_segment_ids=segment_ids_src,
        tgt_input_ids=input_ids_tgt,tgt_input_mask=input_mask_tgt,tgt_labels=labels_tgt)

    
    return feature


def file_based_input_fn_builder(input_file, max_seq_length_src,max_seq_length_tgt, is_training,
                                drop_remainder, is_distributed=False):


    name_to_features = {
        "src_input_ids": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "src_input_mask": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "src_segment_ids": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "tgt_input_ids": tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
        "tgt_input_mask": tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
        "tgt_labels" : tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
        
        
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        #print(example)
        #print(example.keys())

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))
        else:
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))

        return d
    return input_fn
  
  
def get_dataset(processor,
                tokenizer,
                data_dir,
                max_seq_length_src,
                max_seq_length_tgt,
                batch_size,
                mode,
                output_dir,
                is_distributed=False):
    

    if mode == 'train':
        #train_examples = processor.get_train_examples(data_dir)
        #train_file = os.path.join(output_dir, "train.tf_record")
        train_file = "/var/scratch/vro220/train4.tf_record"
        #file_based_convert_examples_to_features(
        #    train_examples, max_seq_length_src,max_seq_length_tgt,
        #    tokenizer, train_file)
        dataset = file_based_input_fn_builder(
            input_file=train_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt =max_seq_length_tgt,
            is_training=True,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})
    elif mode == 'eval':
        #eval_examples = processor.get_dev_examples(data_dir)
        #eval_file = os.path.join(output_dir, "eval.tf_record")
        eval_file = "/var/scratch/vro220/eval4.tf_record"
        #file_based_convert_examples_to_features(
        #    eval_examples, max_seq_length_src,max_seq_length_tgt,
        #    tokenizer, eval_file)
        dataset = file_based_input_fn_builder(
            input_file=eval_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt =max_seq_length_tgt,
            is_training=False,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})
    elif mode == 'test':
      
        #test_examples = processor.get_test_examples(data_dir)
        #test_file = os.path.join(output_dir, "predict.tf_record")
        test_file = "/var/scratch/vro220/predict4.tf_record"
        #file_based_convert_examples_to_features(
        #    test_examples, max_seq_length_src,max_seq_length_tgt,
        #    tokenizer, test_file)
        dataset = file_based_input_fn_builder(
            input_file=test_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt =max_seq_length_tgt,
            is_training=False,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})
    return dataset


class InputFeatures2():
    def __init__(self, src_input_ids,src_input_mask,src_segment_ids):
        self.src_input_ids = src_input_ids
        self.src_input_mask = src_input_mask
        self.src_segment_ids = src_segment_ids

def convert_single_example2(ex_index, example, max_seq_length_src,max_seq_length_tgt,
                           tokenizer):
    tokens_a = tokenizer.tokenize(example['src_txt'])
    
    if len(tokens_a) > max_seq_length_src - 2:
            tokens_a = tokens_a[0:(max_seq_length_src - 2)]

    
    tokens_src = []
    segment_ids_src = []
    tokens_src.append("[CLS]")
    segment_ids_src.append(0)
    for token in tokens_a:
        tokens_src.append(token)
        segment_ids_src.append(0)
    tokens_src.append("[SEP]")
    segment_ids_src.append(0)
  

    input_ids_src = tokenizer.convert_tokens_to_ids(tokens_src)
    
    input_mask_src = [1] * len(input_ids_src)
    
    while len(input_ids_src) < max_seq_length_src:
        input_ids_src.append(0)
        input_mask_src.append(0)
        segment_ids_src.append(0)


    feature = InputFeatures2( src_input_ids=input_ids_src,src_input_mask=input_mask_src,src_segment_ids=segment_ids_src)

    return feature
