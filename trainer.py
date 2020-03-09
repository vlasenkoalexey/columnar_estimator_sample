from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import logging

import os
import tempfile

import time
import math
import pandas as pd
import tensorflow as tf
import google.cloud.logging
import argparse
import datetime

from tensorflow.python.framework import dtypes
from tensorflow.python.data.ops import dataset_ops
from google.cloud import bigquery

BATCH_SIZE = 128
EPOCHS = 5
PROFILER = False
SLOPPY = False

SMALL_TRAIN_DATASET_SIZE = 366715  # select count(1) from `alekseyv-scalableai-dev.criteo_kaggle.train_small`

CSV_SCHEMA = [
      bigquery.SchemaField("label", "INTEGER", mode='REQUIRED'),
      bigquery.SchemaField("int1", "INTEGER"),
      bigquery.SchemaField("int2", "INTEGER"),
      bigquery.SchemaField("int3", "INTEGER"),
      bigquery.SchemaField("int4", "INTEGER"),
      bigquery.SchemaField("int5", "INTEGER"),
      bigquery.SchemaField("int6", "INTEGER"),
      bigquery.SchemaField("int7", "INTEGER"),
      bigquery.SchemaField("int8", "INTEGER"),
      bigquery.SchemaField("int9", "INTEGER"),
      bigquery.SchemaField("int10", "INTEGER"),
      bigquery.SchemaField("int11", "INTEGER"),
      bigquery.SchemaField("int12", "INTEGER"),
      bigquery.SchemaField("int13", "INTEGER"),
      bigquery.SchemaField("cat1", "STRING"),
      bigquery.SchemaField("cat2", "STRING"),
      bigquery.SchemaField("cat3", "STRING"),
      bigquery.SchemaField("cat4", "STRING"),
      bigquery.SchemaField("cat5", "STRING"),
      bigquery.SchemaField("cat6", "STRING"),
      bigquery.SchemaField("cat7", "STRING"),
      bigquery.SchemaField("cat8", "STRING"),
      bigquery.SchemaField("cat9", "STRING"),
      bigquery.SchemaField("cat10", "STRING"),
      bigquery.SchemaField("cat11", "STRING"),
      bigquery.SchemaField("cat12", "STRING"),
      bigquery.SchemaField("cat13", "STRING"),
      bigquery.SchemaField("cat14", "STRING"),
      bigquery.SchemaField("cat15", "STRING"),
      bigquery.SchemaField("cat16", "STRING"),
      bigquery.SchemaField("cat17", "STRING"),
      bigquery.SchemaField("cat18", "STRING"),
      bigquery.SchemaField("cat19", "STRING"),
      bigquery.SchemaField("cat20", "STRING"),
      bigquery.SchemaField("cat21", "STRING"),
      bigquery.SchemaField("cat22", "STRING"),
      bigquery.SchemaField("cat23", "STRING"),
      bigquery.SchemaField("cat24", "STRING"),
      bigquery.SchemaField("cat25", "STRING"),
      bigquery.SchemaField("cat26", "STRING")
  ]

vocab_size = {
'cat1':	  98,
'cat2':		364,
'cat3':		624,
'cat4':		830,
'cat5':		38,
'cat6':		8,
'cat7':		1764,
'cat8':		60,
'cat9':		3,
'cat10':	1267,
'cat11':	1529,
'cat12':	646,
'cat13':	1356,
'cat14':	23,
'cat15':	1254,
'cat16':	727,
'cat17':	9,
'cat18':	836,
'cat19':	284,
'cat20':	4,
'cat21':	667,
'cat22':	9,
'cat23':	12,
'cat24':	822,
'cat25':	37,
'cat26':	601
}

def transform_row(row_dict):
  #tf.print(row_dict)
  label = row_dict.pop('label')
  row_dict.pop('row_hash') # not used
  return (row_dict, label)

def gzip_reader_fn(filenames):
  return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

FixedLenFeature = tf.io.FixedLenFeature
features = {
  'label': FixedLenFeature([], dtype=tf.int64, default_value=0),
  'row_hash': FixedLenFeature([], dtype=tf.int64, default_value=0),
  'int1_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int2_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int3_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int4_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int5_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int6_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int7_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int8_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int9_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int10_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int11_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int12_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'int13_norm': FixedLenFeature([], dtype=tf.float32, default_value=0),
  'cat1': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat2': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat3': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat4': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat5': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat6': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat7': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat8': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat9': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat10': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat11': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat12': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat13': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat14': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat15': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat16': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat17': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat18': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat19': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat20': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat21': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat22': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat23': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat24': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat25': FixedLenFeature([], dtype=tf.string, default_value=""),
  'cat26': FixedLenFeature([], dtype=tf.string, default_value=""),
}

def get_dataset(table_name):
  global BATCH_SIZE
  global EPOCHS
  global CACHE
  global SLOPPY
  filenames = 'gs://alekseyv-scalableai-dev-public-bucket/criteo_kaggle_from_bq_norm/{table_name}_small_norm_*'.format(table_name = table_name)
  print('sloppy: ' + str(SLOPPY))
  if CACHE:
    return tf.data.experimental.make_batched_features_dataset(
      filenames,
      BATCH_SIZE,
      features,
      reader=gzip_reader_fn,
      sloppy_ordering = SLOPPY
    ).map (transform_row).take(get_training_steps_per_epoch()).cache().repeat(EPOCHS)
  else:
    return tf.data.experimental.make_batched_features_dataset(
        filenames,
        BATCH_SIZE,
        features,
        reader=gzip_reader_fn,
        sloppy_ordering = SLOPPY,
        num_epochs = EPOCHS
    ).map (transform_row)

def get_training_steps_per_epoch():
  return SMALL_TRAIN_DATASET_SIZE // BATCH_SIZE

def get_max_steps():
  global EPOCHS
  return EPOCHS * get_training_steps_per_epoch()

def create_feature_columns():
  real_valued_columns = [
      tf.feature_column.numeric_column(field.name + "_norm", shape=()) \
      for field in CSV_SCHEMA if field.field_type == 'INTEGER' and field.name != 'label'
  ]

  categorical_columns = [
      tf.feature_column.categorical_column_with_hash_bucket(
          field.name, vocab_size[field.name] * 5,
      )
      for field in CSV_SCHEMA if field.field_type == 'STRING' and field.name != 'label'
  ]

  return real_valued_columns + categorical_columns

def train_estimator_linear(model_dir):
  global PROFILER
  global EPOCHS

  logging.info('training for {} steps'.format(get_max_steps()))
  config = tf.estimator.RunConfig().replace(save_summary_steps=10)

  profiler_hook = tf.train.ProfilerHook(
      save_steps=get_training_steps_per_epoch(),
      output_dir=os.path.join(model_dir, "profiler"),
      show_dataflow=True,
      show_memory=True)

  hooks = []
  if PROFILER:
    hooks.append(profiler_hook)

  feature_columns = create_feature_columns()
  estimator = tf.estimator.LinearClassifier(
      feature_columns=feature_columns,
      #optimizer=tf.train.FtrlOptimizer(learning_rate=0.0001),
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001),
      model_dir=model_dir,
      config=config
  )
  logging.info('training and evaluating linear estimator model')
  tf.estimator.train_and_evaluate(
      estimator,
      train_spec=tf.estimator.TrainSpec(input_fn=lambda: get_dataset('train'), max_steps=get_max_steps(), hooks=hooks),
      eval_spec=tf.estimator.EvalSpec(input_fn=lambda: get_dataset('test')))
  logging.info('done evaluating estimator model')

def train_estimator(model_dir):
  logging.info('training for {} steps'.format(get_max_steps()))
  config = tf.estimator.RunConfig()
  feature_columns = create_feature_columns()
  estimator = tf.estimator.DNNClassifier(
      optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.005),
      feature_columns=feature_columns,
      hidden_units=[512, 256],
      model_dir=model_dir,
      config=config,
      n_classes=2)
  logging.info('training and evaluating estimator model')
  tf.estimator.train_and_evaluate(
      estimator,
      train_spec=tf.estimator.TrainSpec(input_fn=lambda: get_dataset('train'), max_steps=get_max_steps()),
      eval_spec=tf.estimator.EvalSpec(input_fn=lambda: get_dataset('test')))
  logging.info('done evaluating estimator model')

def run_reader_benchmark(_):
  global EPOCHS
  global BATCH_SIZE
  num_iterations = get_max_steps()
  dataset = get_dataset('train')
  itr = tf.compat.v1.data.make_one_shot_iterator(dataset)
  start = time.time()
  n = 0
  mini_batch = 100
  for _ in range(num_iterations // mini_batch):
    local_start = time.time()
    start_n = n
    for _ in range(mini_batch):
      n += BATCH_SIZE
      _ = itr.get_next()
    local_end = time.time()
    print('Processed %d entries in %f seconds. [%f] examples/s' % (
        n - start_n, local_end - local_start,
        (mini_batch * BATCH_SIZE) / (local_end - local_start)))
  end = time.time()
  print('Processed %d entries in %f seconds. [%f] examples/s' % (
      n, end - start,
      n / (end - start)))

def get_args():
    """Define the task arguments with the default values.
    Returns:
        experiment parameters
    """
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--batch-size',
        help='Batch size for each training and evaluation step.',
        type=int,
        default=512)

    args_parser.add_argument(
        '--num-epochs',
        help='Maximum number of training data epochs on which to train.',
        default=5,
        type=int)

    args_parser.add_argument(
        '--job-dir',
        help='folder or GCS location to write checkpoints and export models.',
        required=True)

    args_parser.add_argument(
        '--startup-function',
        help='Function name to execute when program is started.',
        choices=['train_estimator_linear', 'train_estimator', 'run_reader_benchmark'],
        default='train_estimator_linear')

    args_parser.add_argument(
        '--cache',
        action='store_true',
        help='Cache dataset between epochs.',
        default=False)

    args_parser.add_argument(
        '--sloppy',
        action='store_true',
        help='sloppy_ordering parameter for make_batched_features_dataset.',
        default=False)

    args_parser.add_argument(
        '--docker-file-name',
        help='Ignored by this script')

    args_parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Ignored by this script.',
        default=False)

    args_parser.add_argument(
        '--profiler',
        action='store_true',
        help='Ignored by this script.',
        default=False)

    return args_parser.parse_args()

def main():
    print('running main')
    global BATCH_SIZE
    global EPOCHS
    global PROFILER
    global CACHE
    global SLOPPY
    args = get_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info('trainer called with following arguments:')
    logging.info(' '.join(sys.argv))
    print('setup logging')

    logging.warning('tf version: ' + tf.version.VERSION)

    logging.info('startup_function arg: ' + str(args.startup_function))
    startup_function = getattr(sys.modules[__name__], args.startup_function)

    model_dir = args.job_dir
    logging.info('Model will be saved to "%s..."', model_dir)

    BATCH_SIZE = args.batch_size
    EPOCHS = args.num_epochs
    PROFILER = args.profiler
    CACHE = args.cache
    SLOPPY = args.sloppy

    train_start_time = datetime.datetime.now()

    startup_function(model_dir)

    logging.info('total train time including evaluation: (hh:mm:ss.ms) {}'.format(datetime.datetime.now() - train_start_time))

if __name__ == '__main__':
    main()
