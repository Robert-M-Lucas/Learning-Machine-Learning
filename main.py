import tensorflow as tf

print(f"Tensorflow {tf.version.VERSION}")

if tf.test.gpu_device_name():
    print(f'Default GPU Device: {tf.test.gpu_device_name()}')
else:
    print("Please install GPU version of TF")

string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

# rank1_tensor = tf.Variable(["Test"], tf.string)
# rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
#
# print(tf.rank(rank2_tensor))
# print(tf.rank(rank1_tensor))
# print(tf.rank(string))
# print(rank2_tensor.shape)
# print(tf.shape(rank2_tensor))

ones = tf.ones([3, 2])
print(ones)
flat_ones = tf.reshape(ones, [6])
print(flat_ones)

print(flat_ones.numpy())
