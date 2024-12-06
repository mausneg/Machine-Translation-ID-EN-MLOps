import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import os
import pickle as pkl
from tuner import input_fn, masked_accuracy, masked_loss, INPUT_VOCAB_SIZE, TARGET_VOCAB_SIZE, INPUT_MAX_LEN, TARGET_MAX_LEN
from my_transformer import Transformer

EPOCHS = 50
BATCH_SIZE = 16

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

def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"),
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec,
        )

        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def build_model(hp):
    model = Transformer(
        num_layers=hp['num_layers'],
        num_heads=hp['num_heads'],
        dff=hp['dff'],
        d_model=hp['d_model'],
        input_vocab_size=INPUT_VOCAB_SIZE,
        target_vocab_size=TARGET_VOCAB_SIZE,
        dropout_rate=hp['dropout_rate']
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate']),
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    return model

def run_fn(fn_args: FnArgs):
    hp = fn_args.hyperparameters['values']

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

    with open(os.path.join(fn_args.model_run_dir, 'input_tokenizer.pkl'), 'wb') as f:
        pkl.dump(input_tokenizer, f)
    
    with open(os.path.join(fn_args.model_run_dir, 'target_tokenizer.pkl'), 'wb') as f:
        pkl.dump(target_tokenizer, f)

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

    log_dir = os.path.join(fn_args.model_run_dir, 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        update_freq='batch'
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.01, 
        patience=40,
        restore_best_weights=True
    )

    model = build_model(hp)
    model.fit(
        x=(train_input, train_target_in),
        y=train_target_out,
        validation_data=((eval_input, eval_target_in), eval_target_out),
        epochs=10,
        batch_size=BATCH_SIZE,
        callbacks=[tensorboard_callback, early_stopping_callback]
    )
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples'))
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
