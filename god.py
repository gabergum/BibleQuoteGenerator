import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib


model = load_model("model")
char_to_ind = joblib.load('char_to_ind.pkl')
ind_to_char = joblib.load('ind_to_char.pkl')

def generate_text(model,start_seed,gen_size=500,temp=1.0):
  
  num_generate = gen_size
  
  input_eval = [char_to_ind[s] for s in start_seed]

  input_eval = tf.expand_dims(input_eval,0)

  text_generated = []

  temperature = temp 

  model.reset_states()

  for i in range(num_generate):
    
    predictions = model(input_eval)

    predictions = tf.squeeze(predictions,0)

    predictions = predictions/temperature

    predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()

    input_eval = tf.expand_dims([predicted_id],0)

    text_generated.append(ind_to_char[predicted_id])

  return (start_seed+"".join(text_generated))

  print(generate_text(model,"thanks",gen_size=1000))