from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from urllib.request import urlretrieve
import tarfile
from collections import defaultdict

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
    class_counts = {name: 0 for name in class_names}

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
            if all(count >= samples_per_class for count in class_counts.values()):
                break

        # If we have enough samples, stop processing batches
        if all(count >= samples_per_class for count in class_counts.values()):
            break

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

airplane = balanced_cifar['airplane'][1]

# Display the image
plt.figure(figsize=(3, 3))
plt.imshow(airplane)
plt.axis('off')
plt.show()


#konstanta za broj binova
NUM_BINS = 8

def calculate_normalized_bins_histograms(image_path):
    image = Image.open(image_path)

    #pretvaramo sliku u RGB format(ako nije vec u tom formatu)
    image = image.convert('RGB')

    width, height = image.size

    #dimenzije svakog bina(raspon vrednosti u jednom binu, koliki interval gledamo u odnosu na broj binova)
    bin_size = 256 // NUM_BINS

    #inicijalizujemo histograme za R, G i B komponente
    r_hist = np.zeros(NUM_BINS, dtype=np.float32)
    g_hist = np.zeros(NUM_BINS, dtype=np.float32)
    b_hist = np.zeros(NUM_BINS, dtype=np.float32)

    #iteracija kroz piksele slike
    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))

            #izracunavamo indeks odgovarajuceg bina
            r_bin = r // bin_size
            g_bin = g // bin_size
            b_bin = b // bin_size

            #povecavamo vrednost u odgovarajucem binu
            r_hist[r_bin] += 1
            g_hist[g_bin] += 1
            b_hist[b_bin] += 1

    #normalizacija histograma(podela sa ukupnim brojem piksela)
    total_pixels = width * height
    r_hist /= total_pixels
    g_hist /= total_pixels
    b_hist /= total_pixels

    #vracamo rezultat kao numpy matricu
    return np.stack([r_hist, g_hist, b_hist], axis=0)



def plot_histograms(histograms, bins_num):
    colors = ['red', 'green', 'blue']  #boje za svaku komponentu
    labels = ['Red', 'Green', 'Blue']  #oznake za komponente

    #petlja kroz tri komponente(R, G, B)
    for i in range(3):
        plt.plot(histograms[i], color=colors[i], label=f'{labels[i]} Component')

    #plt.ylim(0, 1)  #Y-osa od 0 do 1
    plt.title('Normalized Color Histograms')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


    print(f"BINS: ")
    bin_width = 256 // bins_num
    for i in range(bins_num):
        start = i * bin_width
        end = (i + 1) * bin_width
        print(f"\t Bin {i}: {start}-{end}")

    print("Normalized Histograms (RGB):")
    for i in range(3):
        print(f"\t {labels[i]} histogram:  {histograms[i]}")





#TREBA ISPRAVITI DA U KODU NEMAMO LINALG I DOT I SKALARNI PROIZVOD
def cosine_similarity(hist1, hist2):
    # flattenovanje histogram matrica u 1D nizove
    flat_hist1 = hist1.flatten()
    flat_hist2 = hist2.flatten()

    dot_product = np.dot(flat_hist1, flat_hist2)  #skalarni proizvod histograma(vektora)
    norm1 = np.linalg.norm(flat_hist1)  #duzina prvog vektora
    norm2 = np.linalg.norm(flat_hist2)  #duzina drugog vektora

    #provera zbog deljenja s nulom
    if norm1 == 0 or norm2 == 0:
        #ako je jedan vektor nula, slicnost je 0
        return 0.0

    #kosinusna slicnost
    similarity = dot_product / (norm1 * norm2)
    return similarity





if __name__ == '__main__':
    image_path1 = '/content/auto.png'
    image_path2 = '/content/kocije.jpg'

    hist1 = calculate_normalized_bins_histograms(image_path1)
    hist2 = calculate_normalized_bins_histograms(image_path2)

    plot_histograms(hist1, NUM_BINS)
    plot_histograms(hist2, NUM_BINS)

    similarity = cosine_similarity(hist1, hist2)
    print(f'Kosinusna sličnost između dve slike je: {similarity:.4f}')