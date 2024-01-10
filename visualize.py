from keras.utils import plot_model
from models import model

plot_model(model, to_file="./tmp/compose-model.png", show_shapes=True)
print("visualization complete")