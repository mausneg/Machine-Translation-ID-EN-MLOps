import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_text as tf_text

INPUT_KEY = 'indonesian'
TARGET_KEY = 'english'

contractions = {
        "'re": " are",
        "n't": " not",
        "'s": " is",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
        "'ve": " have",
        "'em": " them",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "won't": "will not",
        "can't": "cannot",
        "couldn't": "could not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "mightn't": "might not",
        "mustn't": "must not",
        "wasn't": "was not",
        "weren't": "were not",
        "isn't": "is not",
        "aren't": "are not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "needn't": "need not",
        "oughtn't": "ought not",
        "shan't": "shall not",        
    }

def apply_contractions(text):
    for contraction, full_form in contractions.items():
        text = tf.strings.regex_replace(text, contraction, full_form)
    return text

def tranform_seq(inputs):
    inputs = tf_text.normalize_utf8(inputs, 'NFKD')
    inputs = tf.strings.lower(inputs)
    inputs = tf.strings.regex_replace(inputs, r"([^ a-z.?!¡,¿'-])", r"")
    inputs = tf.strings.regex_replace(inputs, r"([.?!¡,¿])", r" \1 ")
    inputs = tf.strings.strip(inputs)
    inputs = apply_contractions(inputs)
    return inputs

def preprocessing_fn(inputs):
    outputs = {}
    outputs['input'] = tranform_seq(inputs[INPUT_KEY])
    temp = tranform_seq(inputs[TARGET_KEY])
    outputs['decoder_input'] = tf.concat([tf.fill([tf.shape(temp)[0], 1], '<start>'), temp], 1)
    outputs['decoder_target'] = tf.concat([temp, tf.fill([tf.shape(temp)[0], 1], '<end>')], 1)   
    return outputs
