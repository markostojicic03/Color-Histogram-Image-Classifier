from copy import deepcopy
from functools import reduce

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from urllib.request import urlretrieve
import tarfile
from collections import defaultdict
import ssl
import matplotlib
# Onemogućavamo proveru sertifikata (privremeno rešenje)
ssl._create_default_https_context = ssl._create_unverified_context
matplotlib.use('TkAgg')  # Postavlja Tkinter kao backend za prikaz grafika
plt.figure(figsize=(8, 6))  # Širina 8, visina 6 inča
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 80

#DICT ZA SLIKE
def download_and_extract_cifar10(root='./data'):
    """
    Download and extract CIFAR-10 dataset if it doesn't exist.
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    if not os.path.exists(root):
        os.makedirs(root)

    filepath = os.path.join(root, filename)

    if not os.path.exists(filepath):
        print("Downloading CIFAR-10...")
        urlretrieve(url, filepath)

    extract_path = os.path.join(root, 'cifar-10-batches-py')
    if not os.path.exists(extract_path):
        print("Extracting files...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=root)

    return extract_path


def load_batch(file_path):
    """
    Load a single CIFAR-10 batch file.
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


def load_balanced_cifar10(samples_per_class=100, root='./data', train=True):
    """
    Load a balanced subset of CIFAR-10 images into a dictionary.

    Args:
        samples_per_class (int): Number of samples to load per class
        root (str): Root directory to store/load CIFAR-10 data
        train (bool): Whether to load from training or test set

    Returns:
        dict: Dictionary with class names as keys and lists of numpy arrays (3x32x32) as values
    """
    # Define the class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # Download and extract the dataset if needed
    data_path = download_and_extract_cifar10(root)

    # Initialize dictionary to store images by class
    class_images = defaultdict(list)
    class_counts = reduce(lambda cnt, name: cnt.update({name: 0}) or cnt, class_names, {})
    #class_counts = {name: 0 for name in class_names}

    if train:
        # Load training batches (1-5)
        batch_files = [f'data_batch_{i}' for i in range(1, 6)]
    else:
        # Load test batch
        batch_files = ['test_batch']

    # Process each batch file
    for batch_file in batch_files:
        batch_path = os.path.join(data_path, batch_file)
        batch = load_batch(batch_path)

        # Get images and labels
        images = batch[b'data']
        labels = batch[b'labels']

        # Process each image
        for img, label in zip(images, labels):
            class_name = class_names[label]

            # If we haven't collected enough samples for this class
            if class_counts[class_name] < samples_per_class:
                # Reshape from (3072,) to (32, 32, 3)
                img_reshaped = img.reshape(3, 32, 32).transpose(1, 2, 0)

                # Add to our collection
                class_images[class_name].append(img_reshaped)
                class_counts[class_name] += 1

            # Check if we've collected enough samples for all classes
            if all(map(lambda count: count >= samples_per_class, class_counts.values())):
                break

        # If we have enough samples, stop processing batches
        if all(map(lambda count: count >= samples_per_class, class_counts.values())):
            break
        print("gotovoopo")
    # Convert defaultdict to regular dict
    return dict(class_images)


# Example usage:
"""
# Load 100 images per class from the training set
balanced_cifar = load_balanced_cifar10(samples_per_class=100)

# Access images for a specific class
airplanes = balanced_cifar['airplane']  # List of 100 numpy arrays (3x32x32)

# Print shapes to verify
for class_name, images in balanced_cifar.items():
    print(f"{class_name}: {len(images)} images, shape: {images[0].shape}")
"""



balanced_cifar = load_balanced_cifar10(samples_per_class=100)


airplane = balanced_cifar['airplane'][5]

# Display the image
plt.figure(figsize=(3, 3))
plt.imshow(airplane)
plt.axis('off')
plt.show()

#konstanta za broj binova
NUM_BINS = 8

def calculate_normalized_bins_histograms(imageArray):
    image = Image.fromarray(imageArray)
    # Pretvaramo sliku u RGB format (ako nije vec u tom formatu)
    image = image.convert('RGB')

    width, height = image.size

    # Dimenzije svakog bina
    bin_size = 256 // NUM_BINS

    # Funkcija za obradu pojedinačnog piksela
    def process_pixel(pixel):
        r, g, b = pixel
        return (
            r // bin_size,
            g // bin_size,
            b // bin_size
        )

    # Transformacija svih piksela slike u indekse binova
    pixels = list(image.getdata())
    bin_indices = map(process_pixel, pixels)

    # Funkcija za akumulaciju vrednosti binova
    def accumulate_bins(hist, bin_index):
        r_bin, g_bin, b_bin = bin_index
        hist[0][r_bin] += 1
        hist[1][g_bin] += 1
        hist[2][b_bin] += 1
        return hist

    # Početni histogrami za R, G, i B komponente
    initial_hist = [
        np.zeros(NUM_BINS, dtype=np.float32),
        np.zeros(NUM_BINS, dtype=np.float32),
        np.zeros(NUM_BINS, dtype=np.float32)
    ]

    # Suma preko svih binova uz pomoć reduce
    histograms = reduce(accumulate_bins, bin_indices, initial_hist)

    # Normalizacija histograma
    total_pixels = width * height

    histograms = list(map(lambda hist: hist / total_pixels, histograms))

    # Vraćamo rezultat kao numpy matricu
    return np.stack(histograms, axis=0)

"""
dict["dog"][1]

"""


def calculate_histogram_dict(balanced_cifar):

    def calculateForOneImage(class_name):#ovde se zove calculate za svaku sliku iz klase koja mu je poslata
        images = balanced_cifar[class_name]
        histograms = list(map(calculate_normalized_bins_histograms, images))
        return class_name, histograms


    #return calculateForOneImage(balanced_cifar['dog'])
    return dict(map(calculateForOneImage, balanced_cifar.keys()))


def calculate_average_histogram(class_name, histograms):

    full_array = np.full((3, 8), 0.0)
    avg = reduce(sub_hist, histograms,full_array)
    return div_hist(avg)

def sub_hist(full_array, hist):

    full_array[0] = sub_arr(full_array[0], hist[0])
    full_array[1] = sub_arr(full_array[1], hist[1])
    full_array[2] = sub_arr(full_array[2], hist[2])
    return full_array

def sub_arr(arr1, arr2):
    return np.array(list(map(lambda item: arr1[item] + arr2[item], range(8))))

def div_hist(avg):
    avg[0] = div_arr(avg[0], 100)
    avg[1] = div_arr(avg[1], 100)
    avg[2] = div_arr(avg[2], 100)
    return avg

def div_arr(arr, x):
    return np.array(list(map(lambda item: arr[item] / x, range(8))))

def plot_histograms(histograms, bins_num=NUM_BINS):
    colors = ['red', 'green', 'blue']  #boje za svaku komponentu
    labels = ['Red', 'Green', 'Blue']  #oznake za komponente

    #petlja kroz tri komponente(R, G, B)
    list(map(lambda i: plt.plot(histograms[i], color=colors[i], label=f'{labels[i]} Component'), range(3)))

    #plt.ylim(0, 1)  #Y-osa od 0 do 1
    plt.title('Normalized Color Histograms')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


    print(f"BINS: ")
    bin_width = 256 // bins_num
    def bin(i:int):
        start = i * bin_width
        end = (i + 1) * bin_width
        print(f"\t Bin {i}: {start}-{end}")
    list(map(lambda i: bin(i), range(bins_num)))


    print("Normalized Histograms (RGB):")
    list(map(lambda i: print(f"\t {labels[i]} histogram:  {histograms[i]}"), range(3)))




import numpy as np
from numpy.linalg import norm
#TREBA ISPRAVITI DA U KODU NEMAMO LINALG I DOT I SKALARNI PROIZVOD
def cosine_similarity(hist1, hist2):
    # flattenovanje histogram matrica u 1D nizove
    flat_hist1 = hist1.flatten()
    flat_hist2 = hist2.flatten()

    dot_product = mul_arr(flat_hist1, flat_hist2)
    sum_of_products = reduce(lambda acc, number: acc + number, dot_product)
    norm1 = reduce(lambda acc, number: acc + number ** 2, flat_hist1, 0) ** 0.5
    norm2 = reduce(lambda acc, number: acc + number ** 2, flat_hist2, 0) ** 0.5
    #provera zbog deljenja s nulom
    if norm1 == 0 or norm2 == 0:
        #ako je jedan vektor nula, slicnost je 0
        return 0.0

    #kosinusna slicnost
    similarity = sum_of_products / (norm1 * norm2)
    return similarity

def mul_arr(arr1, arr2):
    return np.array(list(map(lambda item: arr1[item] * arr2[item], range(24))))
    '''''
    # compute cosine similarity
    cosine = np.sum(hist1 * hist2, axis=1) / (norm(hist1, axis=1) * norm(hist2, axis=1))

def mul_arr(arr1, arr2):
    return np.array(list(map(lambda item: arr1[item] * arr2[item], range(24))))

    if cosine.ndim > 0:
        return np.max(cosine)  # Vraća maksimalnu sličnost u nizu
    return cosine  # Vraća samo skalar
    '''''
def image_classifier(image_path, average_histograms_dict):
    # Ovde sam izvukao objekat klase Image preko putanje slike, zatim sam pretvorio taj image u array(zato sto calculate ocekuje argument koji je array) i zatim sam izracunao histogram za tu sliku.
    image = Image.open(image_path)
    image_array = np.array(image)
    histogram_for_image = calculate_normalized_bins_histograms(image_array)
    # Sada je potrebno da poredimo dobijeni histogram sa prosecnim histogramom svake klase koristeci kosinusnu slicnost.
    average_histograms_dict_copy =  deepcopy(average_histograms_dict)
    result_cosine_similarity = map(
        lambda item: (item[0],  cosine_similarity(histogram_for_image, average_histograms_dict_copy.pop(item[0]))),
        list(average_histograms_dict_copy.items())
    )

    most_similar_class, max_similarity = max(result_cosine_similarity, key=lambda x: x[1])
    # Ispis rezultata
    print("Most similar class:", most_similar_class)
    print("Max cosine similarity:", max_similarity)
    return image_path, most_similar_class, max_similarity
    '''
print(hist1)
    print("------------------------------------------------------------")
    print(hist2)
'''


if __name__ == '__main__':
    # Kreiranje rečnika histograma
    histograms_dict = calculate_histogram_dict(balanced_cifar)
    
    histograms_dict_copy = deepcopy(histograms_dict)

    average_histograms = dict(map(
            lambda item: (item[0], calculate_average_histogram(item[0], histograms_dict_copy.pop(item[0]))),
            list(histograms_dict_copy.items())
    ))

    #print(average_histograms)
    image_classifier("../imageResources/primerAvion2.jpg", average_histograms)
    similarity = cosine_similarity(average_histograms['dog'], average_histograms['frog'])
    print(similarity)

    """"
    OVDE SE ISPISUJU HISTOGRAMI POMOCU PLOT_HISTOGRAMS
    
    # Funkcija za obradu svake klase
    def process_class(class_name):
        histograms = histograms_dict[class_name]

        # Mapiranje za plotovanje svih histograma u trenutnoj klasi
        list(map(lambda histogram: plot_histograms(histogram), histograms))

    similarity = cosine_similarity(average_histograms['dog'], average_histograms['frog'])
    print(similarity)
    # Mapiranje preko svih klasa u rečniku
    list(map(process_class, histograms_dict.keys()))

   """""
