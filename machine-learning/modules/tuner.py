from typing import Any, Dict, NamedTuple, Text
import keras_tuner as kt
import tensorflow as tf
from keras_tuner.engine import base_tuner
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import os
from my_transformer import Transformer

EPOCHS = 50
BATCH_SIZE = 16
INPUT_VOCAB_SIZE = 10000
TARGET_VOCAB_SIZE = 10000
INPUT_MAX_LEN = 50
TARGET_MAX_LEN = 50

input_tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=INPUT_VOCAB_SIZE,
    filters='',
    oov_token='<OOV>'
)

target_tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=TARGET_VOCAB_SIZE,
    filters='',
    oov_token='<OOV>'
)

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner), 
    ("fit_kwargs", Dict[Text, Any])
])

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs,
             batch_size=64)->tf.data.Dataset:
    
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
    )
    return dataset


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)

def build_model(hp: kt.HyperParameters):
    d_model = hp.Int('d_model', min_value=128, max_value=512, step=128)
    num_layers = hp.Int('num_layers', min_value=2, max_value=4, step=1)
    num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
    dff = hp.Int('dff', min_value=256, max_value=1024, step=256)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    lr = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])

    model = Transformer(
        num_layers=num_layers,
        num_heads=num_heads,
        dff=dff,
        d_model=d_model,
        input_vocab_size=INPUT_VOCAB_SIZE,
        target_vocab_size=TARGET_VOCAB_SIZE,
        dropout_rate=dropout_rate
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    return model

def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, EPOCHS, BATCH_SIZE)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, EPOCHS, BATCH_SIZE)

    train_input = train_dataset.map(lambda x: x['inputs']).as_numpy_iterator()
    train_target = train_dataset.map(lambda x: x['targets']).as_numpy_iterator()
    eval_input = eval_dataset.map(lambda x: x['inputs']).as_numpy_iterator()
    eval_target = eval_dataset.map(lambda x: x['targets']).as_numpy_iterator()

    train_input = [str(item) for item in train_input]
    train_target = [str(item) for item in train_target]
    eval_input = [str(item) for item in eval_input]
    eval_target = [str(item) for item in eval_target]

    input_tokenizer.fit_on_texts(train_input)
    target_tokenizer.fit_on_texts(train_target)

    train_input = input_tokenizer.texts_to_sequences(train_input)
    train_target = target_tokenizer.texts_to_sequences(train_target)
    eval_input = input_tokenizer.texts_to_sequences(eval_input)
    eval_target = target_tokenizer.texts_to_sequences(eval_target)

    train_target_in = [seq[:-1] for seq in train_target]
    train_target_out = [seq[1:] for seq in train_target]
    eval_target_in = [seq[:-1] for seq in eval_target]
    eval_target_out = [seq[1:] for seq in eval_target]

    train_input = tf.keras.preprocessing.sequence.pad_sequences(train_input, padding='post', maxlen=INPUT_MAX_LEN)
    train_target_in = tf.keras.preprocessing.sequence.pad_sequences(train_target_in, padding='post', maxlen=TARGET_MAX_LEN)
    train_target_out = tf.keras.preprocessing.sequence.pad_sequences(train_target_out, padding='post', maxlen=TARGET_MAX_LEN)
    eval_input = tf.keras.preprocessing.sequence.pad_sequences(eval_input, padding='post', maxlen=INPUT_MAX_LEN)
    eval_target_in = tf.keras.preprocessing.sequence.pad_sequences(eval_target_in, padding='post', maxlen=TARGET_MAX_LEN)
    eval_target_out = tf.keras.preprocessing.sequence.pad_sequences(eval_target_out, padding='post', maxlen=TARGET_MAX_LEN)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.01, 
        patience=40,
        restore_best_weights=True
    )

    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective('val_masked_accuracy', direction='max'),
        max_epochs=EPOCHS,
        factor=3,
        directory=os.path.join(fn_args.working_dir, 'tuner'),
        project_name='transformer_tuning'
    )
    tuner.oracle.max_trials = 10

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': (train_input, train_target_in),
            'y': train_target_out,
            'validation_data': ((eval_input, eval_target_in), eval_target_out),
            'callbacks': [early_stopping_callback]
        }
    )