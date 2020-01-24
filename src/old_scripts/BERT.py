
def create_model(bert_config, is_training, input_ids, mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
  model = modeling.BertModel(
    config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=use_one_hot_embeddings
  )

  output_layer = model.get_sequence_output()
  # output_layer shape is
  if is_training:
    output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
  logits = hidden2tag(output_layer, num_labels)
  # TODO test shape
  logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
  if FLAGS.crf:
    mask2len = tf.reduce_sum(mask, axis=1)
    loss, trans = crf_loss(logits, labels, mask, num_labels, mask2len)
    predict, viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
    return (loss, logits, predict)

  else:
    loss, predict = softmax_layer(logits, labels, num_labels, mask)

    return (loss, logits, predict)

  def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                       num_train_steps, num_warmup_steps, use_tpu,
                       use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
      logging.info("*** Features ***")
      for name in sorted(features.keys()):
        logging.info("  name = %s, shape = %s" % (name, features[name].shape))
      input_ids = features["input_ids"]
      mask = features["mask"]
      segment_ids = features["segment_ids"]
      label_ids = features["label_ids"]
      is_training = (mode == tf.estimator.ModeKeys.TRAIN)
      if FLAGS.crf:
        (total_loss, logits, predicts) = create_model(bert_config, is_training, input_ids,
                                                      mask, segment_ids, label_ids, num_labels,
                                                      use_one_hot_embeddings)

      else:
        (total_loss, logits, predicts) = create_model(bert_config, is_training, input_ids,
                                                      mask, segment_ids, label_ids, num_labels,
                                                      use_one_hot_embeddings)
      tvars = tf.trainable_variables()
      scaffold_fn = None
      initialized_variable_names = None
      if init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        if use_tpu:
          def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:

          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
      logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                     init_string)

      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

      elif mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(label_ids, logits, num_labels, mask):
          predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
          cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels - 1, weights=mask)
          return {
            "confusion_matrix": cm
          }
          #

        eval_metrics = (metric_fn, [label_ids, logits, num_labels, mask])
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
        )
      return output_spec

    return model_fn