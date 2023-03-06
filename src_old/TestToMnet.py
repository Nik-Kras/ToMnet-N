import tensorflow as tf

from ToMnet_N.CharNet import CharNet
from ToMnet_N.PredNet import PredNet
from ToMnet_N.ToMnet import ToMnet

print("-----")
print("Testing Char Net:")

c = CharNet((10, 12, 12, 11), 5, 8)
dummy_data_1 = tf.ones((16, 10, 12, 12, 11))
out = c.call(dummy_data_1)
print("Input data shape: ", dummy_data_1.shape)
print("Layer output shape: ", out.shape)
# print("Layer response on dummy data: ", out)

print("-----")
print("Testing Pred Net:")

p = PredNet(5)
dummy_data_1 = tf.ones((16, 13, 12, 8))
out = p.call(dummy_data_1)
print("Input data shape: ", dummy_data_1.shape)
print("Layer output shape: ", out.shape)
# print("Layer response on dummy data: ", out)

print("-----")
print("Testing ToMnet-N:")

t = ToMnet()
dummy_data_1 = tf.ones((10, 12, 12, 11))
dummy_data_2 = tf.ones((12, 12, 6))
dummy_data_3 = tf.expand_dims(dummy_data_2, axis=0)
dummy_data_4 = tf.repeat(dummy_data_3, repeats=2, axis=-1)[:,:,:,0:11]
dummy_combined = tf.keras.layers.Concatenate(axis=0)([dummy_data_1, dummy_data_4])
dummy_combined =  tf.expand_dims(dummy_combined, axis=0)
dummy_combined = tf.repeat(dummy_combined, repeats=16, axis = 0)
t.predict(dummy_combined)
print("Input data shape: ", dummy_data_1.shape)
print("Layer output shape: ", out.shape)
print("Model summary #1 ", t.summary())
print("Model summary #2 ", t.model().summary())