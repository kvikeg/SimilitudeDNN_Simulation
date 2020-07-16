import sys 
import os
import tempfile
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.layers.core import Activation
from keras.utils.generic_utils import get_custom_objects
from scipy.stats import entropy
from keras.layers import Layer
import keras.activations

from keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sbs

sbs.set()

# get mnist data to test the tranlation
from keras.datasets import mnist

# size of encoding representation
encoding_dim = 32
epsilon = 0.0000000001

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # raise ValueError("here")
        return 0.0 
        #return K.dot(x, self.kernel)

#    def compute_output_shape(self, input_shape):
#        return (input_shape[0], self.output_dim)

class PolyActivation(Activation):
    def __init__(self, activation, **kwargs):
        super(PolyActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'poly'

def poly(x):
    return K.zeros_like(x)

get_custom_objects().update({'poly': PolyActivation(poly)})

def build_poly_encoder():
    input_img = Input(shape=(784,))
    encoded = Dense(encoding_dim, activation=poly)(input_img)
    poly_encoder = Model(input_img, encoded)
    
    poly_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return poly_encoder

# Copied from https://github.com/raghakot/keras-vis/blob/master/vis/utils/utils.py 
def apply_modifications(model, custom_objects=None):
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    loaded_model =None
    model.save(model_path)
    loaded_model = load_model(model_path, custom_objects=custom_objects, compile=False)
    
    os.remove(model_path)
    return loaded_model

class GraphParams:
    def __init__(self, xlabel, ylabel, legend):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend

def plot_results(noise_levels, entropy_vals, entropy_stds, graph_params, filename=None):
    sbs.set_style("whitegrid", {"axes.grid": True})
    plt.figure()
    plt.errorbar(noise_levels, entropy_vals, yerr=entropy_stds, label=graph_params.legend)
    plt.grid(b=True)
    plt.xlabel(graph_params.xlabel)
    plt.ylabel(graph_params.ylabel)
    if graph_params.legend != None:
        plt.legend(numpoints=1, fancybox=True)
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()

def test_single_image(encoder, decoder, rand_img):
    #rand_img = [rand_img]
    #rand_img = np.array(rand_img)
    boo = encoder.predict(rand_img)
    foo = decoder.predict(boo)

    rand_img = rand_img + epsilon
    foo = foo + epsilon
    return entropy(rand_img.flatten(), qk=foo.flatten())

def test_encoders(encoder, decoder, X_test, image_filename):
    noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    num_runs = 100

    entropy_vals = np.zeros(len(noise_levels))
    entropy_stds = np.zeros(len(noise_levels))
    for noise_ind in range(len(noise_levels)):
        rand_entropy = np.zeros(num_runs)

        for i in range(num_runs):
            rand_img_ind = np.random.randint(0, len(X_test)-1, 1)
            rand_img = X_test[rand_img_ind]
            noisy_img = rand_img + noise_levels[noise_ind] * np.random.normal(loc=0.0, scale=1.0, size=rand_img.shape)
            noisy_img = np.clip(noisy_img, 0., 1.)
            rand_entropy[i] = test_single_image(encoder, decoder, noisy_img)
            
        entropy_vals[noise_ind] = np.mean(rand_entropy)
        entropy_stds[noise_ind] = np.std(rand_entropy)
        
    graph_params = GraphParams('Noise Ratio', 'KL Distance', None)
    plot_results(noise_levels, entropy_vals, entropy_stds, graph_params, image_filename)

if len(sys.argv) > 1:
    model_filename = sys.argv[1] + "_encoder"
    amodel = load_model(model_filename)
    decoder = load_model(sys.argv[1] + "_decoder")
else:
    print("Usage: change_model.py <model name>")
    exit(1)
        
#poly_encoder = build_poly_encoder()

amodel.layers[1].activation = poly

# config = amodel.layers[1].get_config()
# config["activation"] = 
# new_layer = amodel.layers[1].from_config(config)
# amodel.layers[1] = new_layer

# amodel = apply_modifications(amodel, custom_objects=get_custom_objects())

print("boom")
#amodel.layers[1] = MyLayer(784)

weights = amodel.layers[1].get_weights()
# new_layer.set_weights(weights)

# TODO: anything else to set here?

# load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalize and flatten into single dimensional vectors
X_train = x_train.astype('float32') / 255. 
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = x_test.astype('float32') / 255.
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

orig_model = load_model(model_filename)

image_filename = 'KL_distance_autoencoder_changed.png'
image_filename_orig = 'KL_distance_autoencoder_orig.png'
test_encoders(amodel, decoder, X_test, image_filename)
test_encoders(orig_model, decoder, X_test, image_filename_orig)
