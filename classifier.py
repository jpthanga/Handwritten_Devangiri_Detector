import tensorflow as tf
import numpy as np

data = np.load('dim_red.npy')
labels = np.load('labels.npy')

inp = tf.convert_to_tensor(data, dtype=tf.float32)
labs = tf.convert_to_tensor(labels, dtype=tf.float32)



#input
input_layer = inp

layer1 = tf.layers.dense(inputs=input_layer,
                         activation=tf.nn.sigmoid,
                         units=50)

# layer2 = tf.layers.dense(inputs=layer1,
#                          activation=tf.nn.sigmoid,
#                          units=250)

layer3 = tf.layers.dense(inputs=layer1,
                         activation=tf.nn.relu,
                         units=36)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer3, labels=labs))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

k = loss

predi = tf.argmax(tf.nn.softmax(layer3),axis=-1)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

while True:
    _, ki = sess.run((train, k))
    print(ki)
    if(ki < 1.6):
        break

pred = sess.run(predi)

print(pred.shape)
print(pred)
# np.save("p",pred)
