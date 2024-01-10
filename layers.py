from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dense
from keras.layers import BatchNormalization, Conv1D, ReLU, Dropout, GlobalMaxPooling1D

def multi_head_attention(chain, num_heads):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=64, value_dim=64)
    query = attention(chain, chain, chain)
    chain = LayerNormalization()(query) 
    return chain

def affine(chain, neurons):
    chain = Dense(neurons, activation='relu')(chain)
    chain = BatchNormalization()(chain)
    return chain

def conv(chain, filters):
    chain = Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu')(chain)
    chain = BatchNormalization()(chain)
    chain = ReLU()(chain)
    chain = Dropout(0.5)(chain)
    return chain
