import tensorflow as tf
# from tensorflow.keras.layers import Conv2D , Flatten ,Dense
# from tensorflow.keras import Model

print(tf.__version__)

class MyModel(Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv1 = Conv2D(64,(3,3),activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(32,activation = 'relu')
        self.d2 = Dense(16,activation = "relu")
        self.d3 = Dense(3,activation = 'softmax')

    def call(self,x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

