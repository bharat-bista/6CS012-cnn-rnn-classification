import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

with tf.device("/GPU:0"):
    a = tf.random.normal((5000, 5000))
    b = tf.random.normal((5000, 5000))
    c = tf.matmul(a, b)

print("Result shape:", c.shape)
print("CUDA/GPU test complete")
