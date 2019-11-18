import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import time
#get data
(train_images , train_labels),(test_images,test_labels) = keras.datasets.mnist.load_data()

#setup model
dense_layers  = [64,128,256,512]
for dense_layer in dense_layers:
    NAME = "{}-dense-{}".format(dense_layer,int(time.time()))
            
            
    tensorboard= TensorBoard(log_dir='mnist shortest
                             log\{}'.format(NAME))
    print(NAME)

    model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(dense_layer,activation = tf.nn.relu),
    keras.layers.Dense(10,activation = tf.nn.softmax)
    ])
    model.compile(optimizer = 'Adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

    #train the model
    model.fit(train_images , train_labels, epochs  = 120,callbacks = [tensorboard])

#evaluate
test_loss, test_acc = model.evaluate(test_images,test_labels)
print("test accuracy : ", test_acc)

#make predictions
predictions = model.predict(test_images)
