# ChromaClass: Functional Color Histogram Image Classifier

## Project Overview
ChromaClass is a specialized image analysis system developed to classify images based on their color distribution patterns. The project focuses on generating 3D RGB color histograms, computing average class profiles, and implementing a classifier based on cosine similarity metrics. 

A primary objective of this project is the strict application of functional programming principles, ensuring the solution is built entirely without explicit loops or standard comprehensions.

---

## Technical Constraints and Methodology
The implementation adheres to a specific functional paradigm to ensure declarative and mathematical code structure:

* Functional Programming Style: The entire solution is implemented without using explicit for or while loops, and without list or dictionary comprehensions.
* Higher-Order Functions: The logic relies exclusively on lambda functions, map, and reduce for all data processing and aggregation tasks.
* Custom Core Logic: Basic operations, such as determining collection lengths, are implemented manually using reduce-based logic rather than built-in functions like len.
* Library Usage: NumPy is utilized for matrix operations and vector flattening, while following the functional constraints for all calculations.

---

## Core System Functionalities

### 1. 3D Color Histogram Generation
The system processes individual images from a provided path to generate normalized histograms for each RGB component.
* Binning: The color space is partitioned into a configurable number of bins, typically ranging from 8 to 16.
* Normalization: Each histogram is normalized by the total pixel count to allow for accurate comparison between images of different resolutions.

### 2. Average Class Profiling
The system processes lists of images categorized into classes (such as landscapes or portraits) to establish a baseline color profile for each category.
* Data Aggregation: Using map and reduce, the system aggregates histograms from all images within a specific class.
* Mean Calculation: The total aggregate is averaged by the number of samples in the class to create a representative 2D matrix profile.

### 3. Cosine Similarity Metrics
A mathematical comparison layer is used to determine the proximity between different color profiles.
* Vectorization: 2D histogram matrices are flattened into 1D arrays to facilitate vector calculations.
* Similarity Calculation: The system computes the cosine similarity using dot products and norms, all handled through functional data pipelines.

### 4. Automated Image Classifier
The final module serves as a classifier that predicts the category of an unknown image.
* Comparison Logic: The histogram of the target image is compared against the average profile of every available class.
* Prediction Output: The system assigns the image to the class with the highest similarity score, returning the image identifier, the predicted class, and the specific similarity value.

---

## Dataset and Evaluation
The system is designed to be evaluated using standard image datasets such as STL-10, CIFAR-10, or Caltech101. For optimal results, the classifier is typically trained on approximately 100 images per category across 4 to 5 distinct classes.

---

## Execution and Submission
This project is part of the Image Analysis curriculum and is managed via GitHub Classroom.
1. All implementation logic must reside within a single Python source file.
2. The repository must include sample images and relevant JSON configurations if required by the workflow.
3. The execution flow follows the functional pipeline from raw image input to final similarity-based classification.

---
Project developed as part of the Parallel Algorithms course curriculum.
