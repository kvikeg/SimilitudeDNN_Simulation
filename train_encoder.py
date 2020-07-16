from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.datasets import mnist
from keras import regularizers
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import seaborn as sbs
import numpy as np
import sys
from scipy.stats import entropy

sbs.set()

# size of encoding representation
encoding_dim = 32
epsilon = 0.0000000001

def build_autoencoder():
    input_img = Input(shape=(784,))
    #encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-3))(input_img)
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    auto_encoder = Model(input_img, decoded)

    sep_encoder = Model(input_img, encoded)
    
    # placeholder for input
    encoded_input = Input(shape=(encoding_dim,))
    # get the last layer of auto_encoder
    decoder_layer = auto_encoder.layers[-1]
    sep_decoder = Model(encoded_input, decoder_layer(encoded_input)) 
    
    auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return (auto_encoder, sep_encoder, sep_decoder)

def show_images(orig_imgs, decoded_imgs, filename=None):
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(orig_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()

def test_and_draw_single_image(encoder, decoder, rand_img, filename=None):
    rand_img = [rand_img]
    rand_img = np.array(rand_img)
    boo = encoder.predict(rand_img)
    foo = decoder.predict(boo)
    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.imshow(rand_img.reshape(28, 28))
    ax = plt.subplot(1, 2, 2)
    plt.imshow(foo.reshape(28, 28))
    plt.gray()
    if filename == None:
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


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize and flatten into single dimensional vectors
    X_train = x_train.astype('float32') / 255. 
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = x_test.astype('float32') / 255.
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    
    # notice that validation is also x_train
    if len(sys.argv) > 1:
        auto_encoder = load_model(sys.argv[1] + "_autoencoder")
        encoder = load_model(sys.argv[1] + "_encoder")
        decoder = load_model(sys.argv[1] + "_decoder")
    else:
        (auto_encoder, encoder, decoder) = build_autoencoder()
        # auto_encoder.fit(X_train, X_train, batch_size=256, epochs=5, shuffle=True, validation_data=(X_test, X_test), verbose=1, callbacks=[TensorBoard(log_dir='C:\\tmp')])
        auto_encoder.fit(X_train, X_train, batch_size=256, epochs=100, shuffle=True, validation_data=(X_test, X_test), verbose=1)
        auto_encoder.save("exp2_autoencoder")
        encoder.save("exp2_encoder")
        decoder.save("exp2_decoder")


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
    plot_results(noise_levels, entropy_vals, entropy_stds, graph_params, 'KL_distance_autoencoder.png')
    
    # show results 
    #encoded_imgs = encoder.predict(X_test)
    # decoded_imgs = decoder.predict(encoded_imgs)

    # show_images(X_test[:10], decoded_imgs, "auto_encoder_test_imgs.png")

    print("bye bye")