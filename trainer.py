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
import json

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.client import timeline
from google.cloud import bigquery

from tensorflow.python.platform import gfile
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.ops import io_ops
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import parsing_ops
from tensorflow.python.data.experimental.ops import shuffle_ops

ARGS = None
SMALL_TRAIN_DATASET_SIZE = 366715  # select count(1) from `alekseyv-scalableai-dev.criteo_kaggle.train_small`
FULL_TRAIN_DATASET_SIZE = 3500000  # approximate

if tf.version.VERSION.startswith('2'):
  GradientDescentOptimizer = tf.keras.optimizers.SGD
  RunOptions = tf.compat.v1.RunOptions
  RunMetadata = tf.compat.v1.RunMetadata
else:
  GradientDescentOptimizer = tf.train.GradientDescentOptimizer
  RunOptions = tf.RunOptions
  RunMetadata = tf.RunMetadata

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

def transform_row(row_dict):
  label = row_dict.pop('label')
  row_dict.pop('row_hash') # not used
  return (row_dict, label)

def parse_and_transform(tfrecord):
  example = tf.io.parse_example(tfrecord, features)
  transformed_example = transform_row(example)
  return transformed_example

def get_dataset(table_name):
  global ARGS
  filenames = 'gs://alekseyv-scalableai-dev-public-bucket/criteo_kaggle_from_bq_norm/{table_name}{dataset_size}_norm_*'.format(
    table_name = table_name,
    dataset_size = '_small' if ARGS.dataset_size == 'small' else '')

  dataset_function = getattr(sys.modules[__name__], ARGS.dataset_function)
  return dataset_function(filenames)

def make_batched_features_dataset(filenames):
  def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

  if ARGS.cache:
    return tf.data.experimental.make_batched_features_dataset(
      filenames,
      ARGS.batch_size,
      features,
      reader=gzip_reader_fn,
      sloppy_ordering = ARGS.sloppy,
      reader_num_threads=ARGS.reader_num_threads,
      parser_num_threads=ARGS.parser_num_threads
    ).map (transform_row).take(get_training_steps_per_epoch()).cache().repeat(ARGS.num_epochs)
  else:
    return tf.data.experimental.make_batched_features_dataset(
        filenames,
        ARGS.batch_size,
        features,
        reader=gzip_reader_fn,
        sloppy_ordering = ARGS.sloppy,
        num_epochs = ARGS.num_epochs,
        reader_num_threads=ARGS.reader_num_threads,
        parser_num_threads=ARGS.parser_num_threads
    ).map (transform_row)

def manual_new_parallel_inteleave(filenames):
  options = tf.data.Options()
  options.experimental_deterministic = not(ARGS.sloppy)
  filenames_list = gfile.Glob(filenames)
  files_dataset = dataset_ops.Dataset.from_tensor_slices(filenames_list).shuffle(len(filenames_list))

  dataset = files_dataset.with_options(options).interleave(
      lambda file_name: tf.data.TFRecordDataset(file_name, compression_type="GZIP"),
      cycle_length = ARGS.reader_num_threads,
      num_parallel_calls=ARGS.reader_num_threads) \
    .shuffle(10000) \
    .repeat(ARGS.num_epochs) \
    .batch(ARGS.batch_size) \
    .map(parse_and_transform, num_parallel_calls=ARGS.parser_num_threads) \

  if ARGS.cache:
    dataset = dataset.cache()
  return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def manual_old_parallel_inteleave(filenames):
  filenames_list = gfile.Glob(filenames)
  files_dataset = dataset_ops.Dataset.from_tensor_slices(filenames_list).shuffle(len(filenames_list))

  dataset = files_dataset.apply(
      interleave_ops.parallel_interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type="GZIP"),
        cycle_length=ARGS.reader_num_threads,
        sloppy=ARGS.sloppy)) \
    .shuffle(10000) \
    .repeat(ARGS.num_epochs) \
    .batch(ARGS.batch_size) \
    .map(parse_and_transform, num_parallel_calls=ARGS.parser_num_threads) \

  if ARGS.cache:
    dataset = dataset.cache()
  return dataset.prefetch(tf.data.experimental.AUTOTUNE)

def get_training_steps_per_epoch():
  train_dataset_size = SMALL_TRAIN_DATASET_SIZE if ARGS.dataset_size == 'small' else FULL_TRAIN_DATASET_SIZE
  return train_dataset_size // ARGS.batch_size

def get_max_steps():
  global ARGS
  return ARGS.num_epochs * get_training_steps_per_epoch()

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
  global ARGS

  logging.info('training for {} steps'.format(get_max_steps()))
  config = tf.estimator.RunConfig().replace(save_summary_steps=10)

  hooks = []
  if ARGS.profiler:
    profiler_hook = tf.estimator.ProfilerHook(
    save_steps=get_training_steps_per_epoch(),
    output_dir=os.path.join(model_dir, "profiler"),
    show_dataflow=True,
    show_memory=True)
    hooks.append(profiler_hook)

  feature_columns = create_feature_columns()
  estimator = tf.estimator.LinearClassifier(
      feature_columns=feature_columns,
      optimizer=GradientDescentOptimizer(learning_rate=0.001),
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

def run_reader_benchmark(model_dir):
  global ARGS
  tf.compat.v1.disable_eager_execution()
  num_iterations = get_max_steps()
  dataset = get_dataset('train')
  itr = tf.compat.v1.data.make_one_shot_iterator(dataset)
  start = time.time()
  n = 0
  mini_batch = 100
  size = tf.shape(itr.get_next()[0]['cat1'])
  run_options = RunOptions(trace_level=RunOptions.FULL_TRACE)
  run_metadata= RunMetadata()
  ops_metadata = []
  batch_size = ARGS.batch_size
  profiler = ARGS.profiler
  profile_example_start = ARGS.profile_example_start
  profile_example_end = ARGS.profile_example_end

  with tf.compat.v1.Session() as sess:
    start = time.time()
    n = 0
    i = 0
    for _ in range(num_iterations // mini_batch):
      local_start = time.time()
      start_n = n
      for _ in range(mini_batch):
        n += batch_size
        if profiler and i >= profile_example_start and i < profile_example_end:
          sess.run(size, options=run_options, run_metadata=run_metadata)
          ops_metadata.append(run_metadata.step_stats)
        else:
          sess.run(size)
        i += 1
      local_end = time.time()
      logging.info('Processed %d entries in %f seconds. [%f] examples/s' % (
          n - start_n, local_end - local_start,
          (mini_batch * batch_size) / (local_end - local_start)))
    end = time.time()
    logging.info('Processed %d entries in %f seconds. [%f] examples/s' % (
        n, end - start,
        n / (end - start)))

    if ARGS.profiler:
      if ARGS.profiler_combine_traces:
        logging.info('Combinding trace events')
        all_trace_events = []
        for op_metadata in ops_metadata:
          tl = timeline.Timeline(op_metadata)
          ctf = tl.generate_chrome_trace_format()
          trace_events = json.loads(ctf)['traceEvents']
          all_trace_events.extend(trace_events)

        logging.info('Dumping trace events to disk')
        with open(os.path.join(model_dir, 'timeline.json'), 'w') as f:
          f.write(json.dumps({'traceEvents' : all_trace_events}).replace('\n', ' '))
      else:
        logging.info('Dumping trace events to disk')
        i = 0
        for op_metadata in ops_metadata:
          tl = timeline.Timeline(op_metadata)
          ctf = tl.generate_chrome_trace_format().replace('\n', ' ')
          with open(os.path.join(model_dir, 'timeline_%d.json' % i), 'w') as f:
            f.write(ctf)
          i += 1

    logging.info('model dir: %s' % model_dir)

def run_reader_benchmark_eager_mode(_):
  ops.enable_eager_execution()
  global ARGS
  batch_size = ARGS.batch_size
  num_iterations = get_max_steps()
  dataset = get_dataset('train')
#  itr = tf.compat.v1.data.make_one_shot_iterator(dataset)
  start = time.time()
  n = 0
  for row in dataset.take(num_iterations):
    if(row): n += batch_size
  end = time.time()
  logging.info('Processed %d entries in %f seconds. [%f] examples/s' % (
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
        choices=['train_estimator_linear', 'train_estimator', 'run_reader_benchmark', 'run_reader_benchmark_eager_mode'],
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
        '--reader-num-threads',
        help='Number of reader threads, defaults to 1 which is same as default for tf.data.experimental.make_batched_features_dataset',
        type=int,
        default=1)

    args_parser.add_argument(
        '--parser-num-threads',
        help='Number of parser threads, defaults to 2 which is same as default for tf.data.experimental.make_batched_features_dataset',
        type=int,
        default=2)

    args_parser.add_argument(
        '--dataset-function',
        help='Function name that returns dataset.',
        choices=['make_batched_features_dataset', 'manual_new_parallel_inteleave', 'manual_old_parallel_inteleave'],
        default='make_batched_features_dataset')

    args_parser.add_argument(
        '--dataset-size',
        help='Dataset size to use',
        choices=['small', 'full'],
        default='small')

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
        help='Whether to attach profiler.',
        default=False)

    args_parser.add_argument(
        '--profiler-combine-traces',
        action='store_true',
        help='Whether to combine profiler traces.',
        default=False)

    args_parser.add_argument(
        '--profile-example-start',
        help='Index of first example to profile',
        type=int,
        default=0)

    args_parser.add_argument(
        '--profile-example-end',
        help='Index of last example to profile',
        type=int,
        default=20)

    return args_parser.parse_args()

def main():
    global ARGS
    ARGS = get_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info('trainer called with following arguments:')
    logging.info(' '.join(sys.argv))
    logging.info('setup logging')

    logging.warning('tf version: ' + tf.version.VERSION)

    logging.info('startup_function arg: ' + str(ARGS.startup_function))
    startup_function = getattr(sys.modules[__name__], ARGS.startup_function)

    model_dir = ARGS.job_dir
    logging.info('Model will be saved to "%s..."', model_dir)

    train_start_time = datetime.datetime.now()
    startup_function(model_dir)
    logging.info('total train time including evaluation: (hh:mm:ss.ms) {}'.format(datetime.datetime.now() - train_start_time))

if __name__ == '__main__':
    main()
