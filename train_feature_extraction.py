import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import pandas as pd
from sklearn.utils import shuffle

# Load traffic signs data.
training_file = "./train.p"

with open(training_file, mode='rb') as f:
    data = pickle.load(f)
    

# Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)

# Define placeholders and resize operation.
sign_names = pd.read_csv('signnames.csv')
nb_classes = 43
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))
labels = tf.placeholder(tf.int64, None)

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
EPOCHS = 50
BATCH_SIZE = 128
rate = 0.001

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])
correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

# TODO: Train and evaluate the feature extraction model.
def evaluate(X_data, y_data, sess):
    num_examples = X_data.shape[0]
    total_accuracy = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, 
                                                           labels: batch_y,
                                                           })
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(init)
    num_examples = X_train.shape[0]
    print("Training...")
    print()
    for epoch in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run([training_operation], feed_dict={x: batch_x, labels: batch_y})
        validation_accuracy = evaluate(X_val, y_val, sess)
        print("{}/{:.3f} ({}s) ".format(epoch, validation_accuracy, time.time()-t0), end=' ', flush=True)
    
# TODO: Run the trained model

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
