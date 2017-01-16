import sys
import numpy as np
import pickle

def get_class_net(num_spks=3):

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation#, MaxoutDense
    from keras.layers.advanced_activations import LeakyReLU
    from keras.optimizers import Adam

    # Add batch normalization: keras.layers.normalization.BatchNormalization()
    model = Sequential()
    model.add(Dense(16, input_shape=(600,)))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(num_spks))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy', 'precision', 'recall'])
    return model

print("Remember architecture is hardcoded!")

model_path = sys.argv[1]
spk_map_path = sys.argv[2]
data_path = sys.argv[3]

print("argument one: NN weights path:", model_path)
print("argument two: spk map path:", spk_map_path)
print("argument three: data path:", data_path)

test_vector = np.load(data_path)

with open(spk_map_path, 'rb') as f:
    spk_mappings = pickle.load(f)

print("Speaker mappings:", spk_mappings)

model = get_class_net()
model.load_weights(model_path)

score = model.predict(test_vector)
print('Softmax output:', score)
for spk, idx in spk_mappings.items():
    if idx == np.argmax(score[0]):
        if spk == 'sk':
            print("***PREDICTION: you sound like Skanda!***")
        else:
            print("***PREDICTION: probably not Skanda!***")

