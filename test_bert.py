import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

def bert_encode(data):
  return tokenizer(data, max_length=140, padding='max_length', truncation=True, return_tensors="tf")["input_ids"]

def my_model():
  bert_model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=1)
  # additional layers would go here
  
  return bert_model

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()


  # Read the data
train_df = pd.read_csv('./data/train.csv')
train, dev = train_test_split(train_df, test_size=0.1, random_state=42)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

train_encoded = bert_encode(train.text.to_list())
dev_encoded = bert_encode(dev.text.to_list())

train_dataset = (
  tf.data.Dataset
  .from_tensor_slices((train_encoded, train.target))
  .shuffle(100)
  .batch(32)
)

dev_dataset = (
  tf.data.Dataset
  .from_tensor_slices((dev_encoded, dev.target))
  .shuffle(100)
  .batch(32)
)

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

with strategy.scope():
  model = my_model()
  adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

  model.compile(
      loss='binary_crossentropy',
      optimizer=adam_optimizer,
      metrics=['accuracy'])

  model.summary()


  history = model.fit(
  train_dataset,
  batch_size=32,
  epochs=3,
  validation_data=dev_dataset,
  verbose=1)
  #callbacks=[tf.keras.callbacks.EarlyStopping(
  #            patience=6,
  #            min_delta=0.05,
  #            baseline=0.7,
  #            mode='min',
  #            monitor='val_accuracy',
  #            restore_best_weights=True,
  #            verbose=1)
  #          ])


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")