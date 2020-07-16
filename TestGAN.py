from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import numpy as np
import collections
from scipy.stats import entropy
import seaborn as sbs
import pdb

sbs.set()

epsilon = 0.0000000001

def generate_sample(gen, batch_size, latent_dim):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    return gen.predict(noise)

def show_images(imgs):
    cols = 8
    rows = (len(imgs) // cols ) + (len(imgs) % cols)
    
    # Rescale images 0 - 1
    imgs = 0.5 * imgs + 0.5

    fig, axs = plt.subplots(rows, cols)
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            if cnt < len(imgs):
                axs[i,j].imshow(imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
            else:
                break
    #fig.savefig("images/%d.png" % epoch)
    plt.show()
    plt.close()

def show_and_tell(imgs, pred):
    for i in range(len(imgs)):
        plt.figure()
        print("Prediction #", i, " is ", pred[i])
        plt.imshow(imgs[i, :, :,0])
        plt.show()
        plt.close()

def get_class_distribution(dataset):
    return collections.Counter(dataset)

# Other ideas for distribution:
# - classifier accuracy on manually labeled examples
# - classifier "correctness"

def get_positive_distribution(imgs, tags, num_types):
    dist = {} 
    for i in range(len(imgs)):
        if tags[i] in dist:
            dist[tags[i]] += sum(imgs[i] >= 0)
        else:
            dist[tags[i]] = sum(imgs[i] >= 0)

    # normalize
    size = len(imgs)
    for key in dist.keys():
        dist[key] = dist[key] / size
    
    return dist

def get_zero_distribution(imgs, tags, num_types):
    dist = {} 
    for i in range(len(imgs)):
        if tags[i] in dist:
            dist[tags[i]] += sum(imgs[i] < -0.9999)
        else:
            dist[tags[i]] = sum(imgs[i] < -0.9999)

    # normalize
    size = len(imgs)
    for key in dist.keys():
        dist[key] = dist[key] / size

    return dist

def normalize_distribution(dataset, target_distribution):
    # now all classes might be present
    for i in target_distribution.keys():
        dataset.setdefault(i, epsilon)

    return dataset

def compare_distributions(dataset_a, dataset_b):
    da = np.zeros(len(dataset_a.keys()))
    da[list(dataset_a.keys())] = list(dataset_a.values())
    db = np.zeros(len(dataset_b.keys()))
    db[list(dataset_b.keys())] = list(dataset_b.values())

    da /= sum(da)
    db /= sum(db)

    ent = entropy(da, qk=db)

    if np.isinf(ent):
        pdb.set_trace()
    
    return ent

def compare_matrix_distributions(dataset_a, dataset_b):
    ent_array = np.zeros(len(dataset_a.keys()))

    for i in range(len(dataset_a.keys())):
        norm_a = dataset_a[i] + epsilon
        norm_b = dataset_b[i] + epsilon
        ent = entropy(norm_a.flatten(), qk=norm_b.flatten())
        ent_array[i] = ent 

    return (np.mean(ent_array), np.std(ent_array))

class GraphParams:
    def __init__(self, xlabel, ylabel, control_legend, ddiff_legend):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.control_legend = control_legend
        self.ddiff_legend = ddiff_legend

def plot_results(sizes, ddiff, dstd, control, graph_params, filename=None):
    plt.figure()
    carr = np.zeros(len(sizes))
    carr.fill(control)
    plt.plot(sizes, carr, label=graph_params.control_legend)
    plt.errorbar(sizes, ddiff, yerr=dstd, label=graph_params.ddiff_legend)
    plt.grid(b=True)
    plt.xlabel(graph_params.xlabel)
    plt.ylabel(graph_params.ylabel)
    plt.legend(loc = 'upper right', numpoints=1, fancybox=True)
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()

def plot_all_results(sizes, ddiff, dstd, pdiff, pstd, zdiff, zstd, control, graph_params, filename=None):
    sbs.set_style("whitegrid", {"axes.grid": True})
    plt.figure()
    carr = np.zeros(len(sizes))
    carr.fill(control)
    plt.plot(sizes, carr, label=graph_params.control_legend)
    plt.errorbar(sizes, ddiff, yerr=dstd, label=graph_params.ddiff_legend)
    plt.errorbar(sizes, pdiff, yerr=pstd, label=graph_params.pdiff_legend)
    plt.errorbar(sizes, zdiff, yerr=zstd, label=graph_params.zdiff_legend)
    plt.grid(b=True)
    plt.xlabel(graph_params.xlabel)
    plt.ylabel(graph_params.ylabel)
    plt.legend(numpoints=1, fancybox=True)
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()
    
def show_images(gan_imgs, orig_imgs, filename=None):
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(gan_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(orig_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    out_file = None
    if len(sys.argv) > 1:
        out_file = sys.argv[1]

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train / 127.5 - 1.
    X_test = X_test / 127.5 - 1.

    y_train_distribution = get_class_distribution(y_train)
    y_test_distribution = get_class_distribution(y_test)
    num_types = len(y_train_distribution)
    control_distribution_difference = compare_distributions(y_train_distribution, y_test_distribution)
    print("Train/test distribution difference:", control_distribution_difference)
    
    train_positive_distribution = get_positive_distribution(X_train, y_train, num_types)
    test_positive_distribution = get_positive_distribution(X_test, y_test, num_types)
    control_positive_difference = compare_matrix_distributions(train_positive_distribution, test_positive_distribution)
    train_zero_distribution = get_zero_distribution(X_train, y_train, num_types)
    train_zero_distribution = get_zero_distribution(X_test, y_test, num_types)
    control_zero_difference = compare_matrix_distributions(train_zero_distribution, train_zero_distribution)

    ext = ".hdf5"
    filename = "gan_model"
    num_runs = 10
    save_sample_images = True
    generator = load_model(filename + "_generator")
    classifier = load_model(filename + "_classifier" + ext)

    if save_sample_images:
        sample_imgs = generate_sample(generator, batch_size=10, latent_dim=100)
        rand_img_ind = np.random.randint(0, len(X_test)-1, 10)
        test_imgs = X_test[rand_img_ind]
        show_images(sample_imgs, test_imgs, 'generator_sample.png')
        sys.exit()

    multiplier = 1000
    sample_sizes = [1* multiplier, 5 * multiplier, 10 * multiplier, 50 * multiplier]
    ddiff = np.zeros(len(sample_sizes))
    dstd = np.zeros(len(sample_sizes))
    pdiff = np.zeros(len(sample_sizes))
    pstd = np.zeros(len(sample_sizes))
    zdiff = np.zeros(len(sample_sizes))
    zstd = np.zeros(len(sample_sizes))

    for sample_size in range(len(sample_sizes)):
        digits_differences = np.zeros(num_runs)
        zero_differences = np.zeros(num_runs)
        positive_differences = np.zeros(num_runs)
        for i in range(len(digits_differences)):
            # Generate a batch of new images
            sample_imgs = generate_sample(generator, batch_size=sample_sizes[sample_size], latent_dim=100)
            pred_classes = classifier.predict_classes(sample_imgs)
            pred_classes_distribution = get_class_distribution(pred_classes)
            pred_classes_distribution = normalize_distribution(pred_classes_distribution, y_train_distribution)

            sample_pos_distribution = get_positive_distribution(sample_imgs, pred_classes, num_types)
            sample_zero_distribution = get_zero_distribution(sample_imgs, pred_classes, num_types)
            
            distribution_difference = compare_distributions(y_train_distribution, pred_classes_distribution)
            digits_differences[i] = distribution_difference

            positive_difference = compare_matrix_distributions(train_positive_distribution, sample_pos_distribution)
            positive_differences[i] = positive_difference[0] 

            zero_difference = compare_matrix_distributions(train_zero_distribution, sample_zero_distribution)
            zero_differences[i] = zero_difference[0]
            
        print("Distribution Difference:", distribution_difference, "\tPositive Difference:", positive_differences, "\tZero Difference:", zero_differences)
        ddiff[sample_size] = np.mean(digits_differences)
        dstd[sample_size] = np.std(digits_differences)
        pdiff[sample_size] = np.mean(positive_differences)
        pstd[sample_size] = np.std(positive_differences)
        zdiff[sample_size] = np.mean(zero_differences)
        zstd[sample_size] = np.std(zero_differences)


    graph_params = GraphParams('Sample Size', 'KL Distance', 'Train-Test Control', 'Digits Distribution')
    graph_params.pdiff_legend = 'Positive Points Distribution'
    graph_params.zdiff_legend = 'Zero Points Distribution'

    plot_all_results(sample_sizes, ddiff, dstd, pdiff, pstd, zdiff, zstd, control_distribution_difference, graph_params, out_file)
    # plot_results(sample_sizes, ddiff, dstd, control_distribution_difference, graph_params, out_file)
    # plot_results(sample_sizes, pdiff, pstd, control_positive_difference[0], graph_params, None)
    # plot_results(sample_sizes, zdiff, zstd, control_zero_difference[0], graph_params, None)
    print("Bye Bye")