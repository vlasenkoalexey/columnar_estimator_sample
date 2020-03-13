# Profiling TensorFlow v1 input pipeline and analyzing its performance using BigQuery

Recently I had to debug a performance issue in TensorFlow V1.x input pipeline. TensorFlow V2.x has a nice tensorboard profiler plugin which is super handy for such situations. Unfortunately it doesn't work for TensorFlow V1.x, and I wasn't able to find a lot of information on debugging performance in TensorFlow V1.x. So I'll share my findings, since a lot of people are still on TF1.x, and this information might be useful. Most of TF1.x code is built using Estimators API. For profiling Estimators there is a tf.estimator.ProfilerHook, which can save traces every N steps or N seconds. It is pretty easy to use:
Sample showing how to train models in TensorFlow 1.14/1.15 using estimator API

```python
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
```

Once you run this code, it'll save bunch of timeline_xxx.json files under model_dir/profiler.
You can open those files using chrome profiler viewer - navigate to chrome://tracing and open one of json files.
Results looks like this:

![Estimator profile](pictures/estimator_profile.png)

