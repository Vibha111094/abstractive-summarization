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
from bert_data_processors import *

app =Flask(__name__)

import sys
#config

decoder_configuration = {
    'dim': 768,
    'num_blocks': 6,
    'multihead_attention': {
        'num_heads': 16,
        'output_dim': 768
    },
    'position_embedder_hparams': {
        'dim': 768
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim=768)
}        




optimizer_config = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9
        }
    }
}



lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (768 ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 15000,
}

end_of_sentence_bert_id = 102

run_mode= "decode"


max_seq_length_src = 512
max_seq_length_tgt = 400

bert_pretrain_dir = 'bert_pretrained_models/uncased_L-12_H-768_A-12'       

bert_config = model_utils.transform_bert_to_texar_config(
            os.path.join(bert_pretrain_dir, 'bert_config.json'))
#print(bert_config)



tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
        do_lower_case=True)

vocab_size = len(tokenizer.vocab)

processor = CNNDailymail()
train_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,4,'train',"./")
eval_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,4,'eval',"./")
test_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,4,'test',"./")

src_input_ids = tf.placeholder(tf.int64, shape=(None, None))
src_segment_ids = tf.placeholder(tf.int64, shape=(None, None))
tgt_input_ids = tf.placeholder(tf.int64, shape=(None, None))
tgt_segment_ids = tf.placeholder(tf.int64, shape=(None, None))

batch_size = tf.shape(src_input_ids)[0]

src_input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(src_input_ids, 0)),
                             axis=1)
tgt_input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(src_input_ids, 0)),
                             axis=1)

labels = tf.placeholder(tf.int64, shape=(None, None))
is_target = tf.to_float(tf.not_equal(labels, 0))


global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')
iterator = tx.data.FeedableDataIterator({
        'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset})

batch = iterator.get_next()
#encoder Bert model
print("Intializing the Bert Encoder Graph")
with tf.variable_scope('bert'):
        embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.vocab_size,
            hparams=bert_config.embed)
        word_embeds = embedder(src_input_ids)

        # Creates segment embeddings for each type of tokens.
        segment_embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.type_vocab_size,
            hparams=bert_config.segment_embed)
        segment_embeds = segment_embedder(src_segment_ids)

        input_embeds = word_embeds + segment_embeds

        # The BERT model (a TransformerEncoder)
        #bert_config.encoder=tf.stop_gradient(bert_config.encoder)
        encoder = tx.modules.TransformerEncoder(hparams=bert_config.encoder)
        #encoder = tf.stop_gradient(encoder)
        encoder_output = encoder(input_embeds, src_input_length)
        print("The output of the encoder is")
        print(encoder_output)
        #encoder_output = tf.stop_gradient(encoder_output)
        
        # Builds layers for downstream classification, which is also initialized
        # with BERT pre-trained checkpoint.
        
        with tf.variable_scope("pooler"):
            # Uses the projection of the 1st-step hidden vector of BERT output
            # as the representation of the sentence
            bert_sent_hidden = tf.squeeze(encoder_output[:, 0:1, :], axis=1)
            bert_sent_output = tf.layers.dense(
                bert_sent_hidden, config_downstream.hidden_dim,
                activation=tf.tanh)
            output = tf.layers.dropout(
                bert_sent_output, rate=0.1, training=tx.global_mode_train())
       

print("loading the bert pretrained weights")
# Loads pretrained BERT model parameters
init_checkpoint = os.path.join(bert_pretrain_dir, 'bert_model.ckpt')
model_utils.init_bert_checkpoint(init_checkpoint)
gm = tf.get_default_graph().get_tensor_by_name('bert/word_embeddings_1/global_mode:0')
#decoder part and mle losss
tgt_embedding = tf.concat(
    [tf.zeros(shape=[1, embedder.dim]), embedder.embedding[1:, :]], axis=0)

decoder = tx.modules.TransformerDecoder(embedding=tgt_embedding,
                             hparams=decoder_configuration)
# For training
outputs = decoder(
    memory=encoder_output,
    memory_sequence_length=src_input_length,
    inputs=embedder(tgt_input_ids),
    sequence_length=tgt_input_length,
    decoding_strategy='train_greedy',
    mode=tf.estimator.ModeKeys.TRAIN
)

loss_label_confidence = 0.9
mle_loss = transformer_utils.smoothing_cross_entropy(
        outputs.logits, labels, vocab_size, loss_label_confidence)
mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)

tvars =tf.trainable_variables()
non_bert_vars = [var for var in tvars if 'bert' not in var.name]

    
train_op = tx.core.get_train_op(
mle_loss,
learning_rate=learning_rate,
variables=non_bert_vars,
global_step=global_step,
hparams=optimizer_config)

tf.summary.scalar('lr', learning_rate)
tf.summary.scalar('mle_loss', mle_loss)
summary_merged = tf.summary.merge_all()

begin_of_sentence_bert_id =101
start_tokens = tf.fill([tx.utils.get_batch_size(src_input_ids)],
                       begin_of_sentence_bert_id)
beam_width=5
predictions = decoder(
    memory=encoder_output,
    memory_sequence_length=src_input_length,
    decoding_strategy='infer_greedy',
    beam_width=beam_width,
    length_penalty=1.0,
    start_tokens=start_tokens,
    end_token=end_of_sentence_bert_id,
    max_decoding_length=350,
    mode=tf.estimator.ModeKeys.PREDICT
)


inferred_ids = predictions['sample_id'][:, :, 0]
saver = tf.train.Saver(max_to_keep=5)


 
import os
def _train_epoch(sess, epoch, step, smry_writer):
        
            
        fetches = {
            'step': global_step,
            'train_op': train_op,
            'smry': summary_merged,
            'loss': mle_loss,
        }

        while True:
            try:
              feed_dict = {
                iterator.handle: iterator.get_handle(sess, 'train'),
                tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
              }
              op = sess.run([batch],feed_dict)
              feed_dict = {
                   src_input_ids:op[0]['src_input_ids'],
                   src_segment_ids : op[0]['src_segment_ids'],
                   tgt_input_ids:op[0]['tgt_input_ids'],

                   labels:op[0]['tgt_labels'],
                   learning_rate: utils.get_lr(step, lr),
                   tx.global_mode(): tf.estimator.ModeKeys.TRAIN
                }


              fetches_ = sess.run(fetches, feed_dict=feed_dict)
              step, loss = fetches_['step'], fetches_['loss']
              display_steps = 100
              if step and step % display_steps == 0:
              #    with open(os.path.join('/var/scratch/vro220','modeltime_check_loss.txt'),'a')as  file_obj:
              #        print(step,loss,file=file_obj)
                  #logger.info('step: %d, loss: %.4f', step, loss)
                  print('step: %d, loss: %.4f' % (step, loss))
                  smry_writer.add_summary(fetches_['smry'], global_step=step)

              if step and step % 1000 == 0:
                  model_path = "/var/scratch/vro220/models10/model_"+str(step)+".ckpt"
                  print('saving model to %s' % model_path)
                  saver.save(sess, model_path)
                 # _eval_epoch(sess, epoch,step,mode='eval')
            except tf.errors.OutOfRangeError:
                break

        return step

                
            
import time  
def _eval_epoch2(sess, epoch, step,mode):
        
        iterator.initialize_dataset(sess,'eval')
        avg = 0
        rouge_1f =0
        rouge_1p = 0
        rouge_1r = 0
        rouge_2f = 0
        rouge_2p = 0
        rouge_2r = 0
        rouge_lf = 0
        rouge_lp = 0
        rouge_lr = 0 

        references, hypotheses,my_input,my_try = [], [], [], []
        fetches= {
                'inferred_ids': inferred_ids
            }

        bno=0
        while (True):
            try:
              print("Batch",bno)
              feed_dict = {
              iterator.handle: iterator.get_handle(sess, 'eval'),
              tx.global_mode(): tf.estimator.ModeKeys.EVAL,
              }
              op = sess.run([batch],feed_dict)
              
              
              feed_dict = {
                   src_input_ids:op[0]['src_input_ids'],
                   src_segment_ids : op[0]['src_segment_ids'],
                   tx.global_mode(): tf.estimator.ModeKeys.EVAL
              }
              #start = time.localtime()
              fetches_= sess.run(fetches, feed_dict=feed_dict)
              #end = time.localtime()
              #print("time taken is")
              #print(end.tm_sec - start.tm_sec)

              
              labels = op[0]['tgt_labels']
             
              
              my_input.extend(m.tolist() for m in op[0]['src_input_ids'] )
              hypotheses.extend(h.tolist() for h in fetches_['inferred_ids'])
              references.extend(r.tolist() for r in labels)
             
                
              hypotheses = utils.list_strip_eos(hypotheses, end_of_sentence_bert_id)
              references = utils.list_strip_eos(references, end_of_sentence_bert_id)
              references = utils.list_strip_eos(references, 0)
              my_input = utils.list_strip_eos(my_input, end_of_sentence_bert_id)
                
              rouge = Rouge()
              
              for i in range(4):
                  myindex = (4*bno)+i
 
                  hwords = tokenizer.convert_ids_to_tokens(hypotheses[myindex])
                  rwords = tokenizer.convert_ids_to_tokens(references[myindex])
                  iwords = tokenizer.convert_ids_to_tokens(my_input[myindex])
                  hwords = tx.utils.str_join(hwords).replace(" ##","")
                  rwords = tx.utils.str_join(rwords).replace(" ##","")
                  
                    
                  iwords = tx.utils.str_join(iwords).replace(" ##","")
                  rouge = Rouge()
                  r = rouge.get_scores(hwords, rwords)

                      
                      
                  print("rougue is")
                  print(r)
                  rouge_1f = rouge_1f+r[0]['rouge-1']['f']
                  rouge_1p = rouge_1p+r[0]['rouge-1']['p']
                  rouge_1r = rouge_1r+r[0]['rouge-1']['r']
                  rouge_2f = rouge_2f+r[0]['rouge-2']['f']
                  rouge_2p = rouge_2p+r[0]['rouge-2']['p']
                  rouge_2r = rouge_2r+r[0]['rouge-2']['r']
                  rouge_lf = rouge_lf+r[0]['rouge-l']['f']
                  rouge_lp = rouge_lp+r[0]['rouge-l']['p']
                  rouge_lr = rouge_lr+r[0]['rouge-l']['r']                   
                  print("score is")
                  print(rouge_1f)
                  print("Original Paragraph",iwords)
                  print("Original",rwords)
                  print("Generated",hwords)
              print(bno)
              bno = bno+1
              
            except tf.errors.OutOfRangeError:
                break
                
        print(rouge_1f)
        print(rouge_1p)
        print(rouge_1r)
        print(rouge_2f) 
        print(rouge_2p) 
        print(rouge_2r) 
        print(rouge_lf) 
        print(rouge_lp) 
        print(rouge_lr) 
        print(bno)
    
        with open(os.path.join('/var/scratch/vro220','final_scores.txt'),'a')as  file_obj:
            print(rouge_1f)
            print(rouge_1p)
            print(rouge_1r)
            print(rouge_2f)
            print(rouge_2p)
            print(rouge_2r)
            print(rouge_lf)
            print(rouge_lp)
            print(rouge_lr)
            print(bno) 


def inference(story,tokenizer,sess):
      references, hypotheses = [], []
      my_story = {"src_txt":story}

      features = convert_single_example2(1,my_story,max_seq_length_src,max_seq_length_tgt,tokenizer)

      feed_dict = {
      src_input_ids:np.array(features.src_input_ids).reshape(1,-1),
      src_segment_ids:np.array(features.src_segment_ids).reshape(1,-1),
      gm:tf.estimator.ModeKeys.PREDICT
      }


      fetches = {
      'inferred_ids': inferred_ids,
      }

      fetches_ = sess.run(fetches, feed_dict=feed_dict)
      hypotheses.extend(h.tolist() for h in fetches_['inferred_ids'])
      hypotheses = utils.list_strip_eos(hypotheses, end_of_sentence_bert_id)
      hwords = tokenizer.convert_ids_to_tokens(hypotheses[0])
      hwords = tx.utils.str_join(hwords).replace(" ##","")

      return hwords

model_dir="/var/scratch/vro220/models11/"
import sys

print (len(sys.argv))
print (str(sys.argv[1]))	

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    smry_writer = tf.summary.FileWriter(model_dir, graph=sess.graph)

    if sys.argv[1] == 'train_and_evaluate':


        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        
        iterator.initialize_dataset(sess)

        step = 5000
        max_train_epoch = 20
        for epoch in range(max_train_epoch):
          iterator.restart_dataset(sess, 'train')
          step = _train_epoch(sess, epoch, step, smry_writer)

    elif sys.argv[1] == 'test':
        
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        _eval_epoch2(sess,0,0, mode='eval')
    
    elif sys.argv[1] == 'decode':
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        story = input('Enter your story: ')
        hwords = inference(story,tokenizer,sess)
        print("summary")
        print(hwords)        

    else:
        raise ValueError('Unknown mode: {}'.format(run_mode))


