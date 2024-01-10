from tensorflow import keras
from keras.models import Sequential, Model
from keras.metrics import Precision, Recall
from keras.layers import Embedding, Concatenate, Dense, Dropout, GlobalMaxPooling1D
from layers import conv, affine, multi_head_attention 

max_words = 10000
max_len = 50
embedding_dim = 64

def branch():
    input_layer = keras.Input(shape=(max_len,))
    emb = Embedding(max_words, embedding_dim, input_length=max_len)(input_layer)
    mha1 = multi_head_attention(emb, 16)
    conv1 = conv(mha1, 128)
    mha2 = multi_head_attention(conv1, 16)
    conv2 = conv(mha2, 64)
    pooling = GlobalMaxPooling1D()(conv2)
    aff = affine(pooling, 64)
    return input_layer, aff

input1, branch1 = branch()
input2, branch2 = branch()

concatenated = Concatenate()([branch1, branch2])
hid_layer = Dense(128, activation='relu')(concatenated)
dropout = Dropout(0.5)(hid_layer)
output_layer = Dense(6, activation='softmax')(dropout)

model = Model(inputs=[input1, input2], outputs=output_layer)

model.compile(optimizer='adamax', loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

model.summary()