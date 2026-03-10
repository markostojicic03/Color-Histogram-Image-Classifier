# Color Histogram Analysis and Classification

## Project Overview
The objective of this project is to analyze color histograms in images categorized into different classes (e.g., landscapes, portraits). The system calculates RGB color histograms for each image, determines average histograms per class, and implements a simple classifier based on cosine similarity. 

The implementation strictly follows a functional programming style, avoiding explicit loops and comprehensions in favor of higher-order functions.

---

## Programming Constraints
The solution is built using specific functional patterns:

* No explicit for/while loops or list/dictionary comprehensions are allowed.
* Built-in functions like len cannot be used for collection size; instead, they must be implemented via reduce.
* The use of lambda functions, map, and reduce is required for data processing.

---

## Core Functionalities

### 1. 3D Color Histogram Calculation
The system generates normalized histograms for the RGB components of an image.
* The number of bins is defined as a global constant, typically between 8 and 16.
* Each histogram is normalized by dividing it by the total number of pixels in the image.

### 2. Average Class Histograms
The system processes pairs of class labels and image paths to compute a representative profile for each category.
* Map and reduce are used to aggregate histograms for all images within a specific class.
* The aggregated result is divided by the number of images in the class to produce the average.

### 3. Cosine Similarity Calculation
To compare histograms, the system calculates their cosine similarity.
* 2D histogram matrices are flattened into 1D arrays.
* Functional pipelines (map/reduce) are used to compute dot products and norms required for the similarity score.

### 4. Histogram-Based Classifier
A classifier predicts an image's class by comparing its histogram to the average histograms of all available classes.
* The image is assigned to the class with the highest cosine similarity.
* The function returns the image identifier, the predicted class, and the similarity value.

---

Developed for the Parallel Algorithms course.
