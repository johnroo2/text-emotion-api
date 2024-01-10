import pickle
import numpy as np
import tensorflow as tf

emotions = {0: 'anger', 1: 'fear', 2: 'joy', 3:'love', 4:'sadness', 5:'suprise'}

def predict(text, model_path, token_path):
    model = tf.keras.models.load_model(model_path)
    
    with open(token_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    sequences = tokenizer.texts_to_sequences([text])
    x_new = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)
    predictions = model.predict([x_new, x_new])
    
    label = list(emotions.values())
    probs = list(predictions[0])
    
    return (label, probs)
    
def predict_root(input):
     return predict(input, 'nlp-prototype.h5', 'tokenizer-prototype.pkl')
    # return predict(input, 'archive/nlp.h5', 'archive/tokenizer.pkl')