# !pip3 install tensorflow-hub==0.4.0

import os
import tensorflow as tf
import numpy as np
import google.datalab.bigquery as bq

# PROJECT = 'cloud-training'
# BUCKET = 'cloud-training-ml'
# REGION = 'us-central1'
#
# os.environ['PROJECT'] = PROJECT
# os.environ['BUCKET'] = BUCKET
# os.environ['REGION'] = REGION
# os.environ['TFVERSION'] = '1.8'

## To start, loading the list of categories, authors and article ids created
categories_list = open("categories.txt").read().splitlines()
authors_list = open("authors.txt").read().splitlines()
content_ids_list = open("content_ids.txt").read().splitlines()

mean_months_since_epoch = 523

# For the embedded_title_column feature column, use a Tensorflow Hub Module to create an embedding of the article title.
# Since the articles and titles are in German, you'll want to use a German language embedding module

embedded_title_column = hub.text_embedding_column(
    key="title", module_spec="https://tfhub.dev/google/nnlm-de-dim50/1", trainable=False
)

content_id_column = tf.feature_column.categorical_column_with_hash_bucket(
    key="content_id", hash_bucket_size=len(content_ids_list) + 1
)
embedded_content_column = tf.feature_column.embedding_column(
    categorical_column=content_id_column, dimension=10
)

author_column = tf.feature_column.categorical_column_with_hash_bucket(
    key="author", hash_bucket_size=len(authors_list) + 1
)
embedded_author_column = tf.feature_column.embedding_column(
    categorical_column=author_column, dimension=3
)

category_column_categorical = tf.feature_column.categorical_column_with_vocabulary_list(
    key="category", vocabulary_list=categories_list, num_oov_buckets=1
)
category_column = tf.feature_column.indicator_column(category_column_categorical)

months_since_epoch_boundaries = list(range(400, 700, 20))
months_since_epoch_column = tf.feature_column.numeric_column(key="months_since_epoch")
months_since_epoch_bucketized = tf.feature_column.bucketized_column(
    source_column=months_since_epoch_column, boundaries=months_since_epoch_boundaries
)

crossed_months_since_category_column = tf.feature_column.indicator_column(
    tf.feature_column.crossed_column(
        keys=[category_column_categorical, months_since_epoch_bucketized],
        hash_bucket_size=len(months_since_epoch_boundaries)
        * (len(categories_list) + 1),
    )
)

# Building up feature columns
feature_columns = [
    embedded_content_column,
    embedded_author_column,
    category_column,
    embedded_title_column,
    crossed_months_since_category_column,
]

# Create the input function for this model
record_defaults = [
    ["Unknown"],
    ["Unknown"],
    ["Unknown"],
    ["Unknown"],
    ["Unknown"],
    [mean_months_since_epoch],
    ["Unknown"],
]
column_keys = [
    "visitor_id",
    "content_id",
    "category",
    "title",
    "author",
    "months_since_epoch",
    "next_content_id",
]
label_key = "next_content_id"


def read_dataset(filename, mode, batch_size=512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=record_defaults)
            features = dict(zip(column_keys, columns))
            label = features.pop(label_key)
            return features, label

        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    return _input_fn


# Create Molde which recommends an article for a visitor to the Kurier.at website
def model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params["feature_columns"])
    for units in params["hidden_units"]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params["n_classes"], activation=None)

    predicted_classes = tf.argmax(logits, 1)
    from tensorflow.python.lib.io import file_io

    with file_io.FileIO("content_ids.txt", mode="r") as ifp:
        content = tf.constant([x.rstrip() for x in ifp])
    predicted_class_names = tf.gather(content, predicted_classes)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "class_ids": predicted_classes[:, tf.newaxis],
            "class_names": predicted_class_names[:, tf.newaxis],
            "probabilities": tf.nn.softmax(logits),
            "logits": logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    table = tf.contrib.lookup.index_table_from_file(vocabulary_file="content_ids.txt")
    labels = table.lookup(labels)
    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predicted_classes, name="acc_op"
    )
    top_10_accuracy = tf.metrics.mean(
        tf.nn.in_top_k(predictions=logits, targets=labels, k=10)
    )

    metrics = {"accuracy": accuracy, "top_10_accuracy": top_10_accuracy}

    tf.summary.scalar("accuracy", accuracy[1])
    tf.summary.scalar("top_10_accuracy", top_10_accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# Train and Evaluate
outdir = "content_based_model_trained"
shutil.rmtree(outdir, ignore_errors=True)  # start fresh each time
tf.summary.FileWriterCache.clear()  # ensure filewriter cache is clear for TensorBoard events file
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=outdir,
    params={
        "feature_columns": feature_columns,
        "hidden_units": [200, 100, 50],
        "n_classes": len(content_ids_list),
    },
)

train_spec = tf.estimator.TrainSpec(
    input_fn=read_dataset("training_set.csv", tf.estimator.ModeKeys.TRAIN),
    max_steps=2000,
)

eval_spec = tf.estimator.EvalSpec(
    input_fn=read_dataset("test_set.csv", tf.estimator.ModeKeys.EVAL),
    steps=None,
    start_delay_secs=30,
    throttle_secs=60,
)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# Make predictions with the trained model
output = list(
    estimator.predict(
        input_fn=read_dataset("first_5.csv", tf.estimator.ModeKeys.PREDICT)
    )
)
import numpy as np

recommended_content_ids = [
    np.asscalar(d["class_names"]).decode("UTF-8") for d in output
]
content_ids = open("first_5_content_ids").read().splitlines()

import google.datalab.bigquery as bq

recommended_title_sql = """
#standardSQL
SELECT
(SELECT MAX(IF(index=6, value, NULL)) FROM UNNEST(hits.customDimensions)) AS title
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,
  UNNEST(hits) AS hits
WHERE
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) = \"{}\"
LIMIT 1""".format(
    recommended_content_ids[0]
)

current_title_sql = """
#standardSQL
SELECT
(SELECT MAX(IF(index=6, value, NULL)) FROM UNNEST(hits.customDimensions)) AS title
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,
  UNNEST(hits) AS hits
WHERE
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) = \"{}\"
LIMIT 1""".format(
    content_ids[0]
)
recommended_title = (
    bq.Query(recommended_title_sql)
    .execute()
    .result()
    .to_dataframe()["title"]
    .tolist()[0]
)
current_title = (
    bq.Query(current_title_sql).execute().result().to_dataframe()["title"].tolist()[0]
)
# print("Current title: {} ".format(current_title))
# print("Recommended title: {}".format(recommended_title))
