# Convert exported graph file into TFLite model file
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('/home/paul/General/Linux/custom_model_lite/saved_model')
tflite_model = converter.convert()

with open('/home/paul/General/Linux/custom_model_lite/detect.tflite', 'wb') as f:
  f.write(tflite_model)