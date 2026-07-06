import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__)

print("\nLoading model...\n")

model = tf.keras.models.load_model(
    "garbage_classifier.keras",
    compile=False
)

print("\n✅ MODEL LOADED SUCCESSFULLY!\n")

model.summary()