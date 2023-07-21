#from torchvision import datasets
import os
import numpy as np
import cv2
from sklearn.neural_network import MLPClassifier
from keras.models import load_model
from keras import backend as K

from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Set the path to save the dataset (change it according to your preference)
#dataset_path = "./stl10_dataset/"

# Download the dataset (if not already downloaded)
#stl10_dataset = datasets.STL10(dataset_path, split='train', download=True)

# Access the data and labels
#X = stl10_dataset.data
#y = stl10_dataset.labels

# Print the shape of the data and labels
#print("Data shape:", X.shape)
#print("Labels shape:", y.shape)


# # Extract Features and building a downstream task classifier (a MLP) with hidden_layer = 3
# 
#


# Set the path to the STL10 dataset directory
dataset_path = "./stl10_dataset/stl10_binary"
train_images_path = os.path.join(dataset_path, "train_X.bin")
train_labels_path = os.path.join(dataset_path, "train_y.bin")
test_images_path = os.path.join(dataset_path, "test_X.bin")
test_labels_path = os.path.join(dataset_path, "test_y.bin")

# Load the dataset files and preprocess the images
def read_images(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
        data = data.reshape(-1, 3, 96, 96)
        data = np.transpose(data, (0, 2, 3, 1))
        return data.astype('float32') / 255.0

# Load the training images and labels
train_images = read_images(train_images_path)
train_labels = np.fromfile(train_labels_path, dtype=np.uint8)

# Load the test images and labels
test_images = read_images(test_images_path)
test_labels = np.fromfile(test_labels_path, dtype=np.uint8)

# Resize the images to the desired shape
desired_shape = (64, 64)
train_images_resized = np.array([cv2.resize(image, desired_shape) for image in train_images])
test_images_resized = np.array([cv2.resize(image, desired_shape) for image in test_images])
# Define the custom loss function with masking
def masked_mse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0.0), K.floatx())
    masked_squared_error = K.square(mask * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error) / K.sum(mask)
    return masked_mse
# Load the saved autoencoder model
autoencoder_model = load_model("autoencoder_BD256_20msk_model.h5", custom_objects={"masked_mse": masked_mse})

# Extract features from the STL10 dataset using the autoencoder model
def extract_features(images, model):
    features = model.predict(images)
    features = features.reshape(features.shape[0], -1)
    return features

# Extract features from the resized images
train_features = extract_features(train_images_resized, autoencoder_model)
test_features = extract_features(test_images_resized, autoencoder_model)

# Create and train the MLP classifier
hidden_layer_sizes = (100, 100, 100)
mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
mlp_classifier.fit(train_features, train_labels)

# Evaluate the performance of the classifier
accuracy = mlp_classifier.score(test_features, test_labels)
print("Accuracy hidden_layer_sizes = (100, 100, 100):", accuracy)


# Create and train the MLP classifier with five hidden layers
hidden_layer_sizes = (100, 100, 100, 100, 100)
mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
mlp_classifier.fit(train_features, train_labels)

# Evaluate the performance of the classifier
accuracy = mlp_classifier.score(test_features, test_labels)
print("Accuracy hidden_layer_sizes = (100, 100, 100, 100, 100):", accuracy)


# Create and train the MLP classifier
hidden_layer_sizes = (200, 200, 200)
mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
mlp_classifier.fit(train_features, train_labels)

# Evaluate the performance of the classifier
accuracy = mlp_classifier.score(test_features, test_labels)
print("Accuracy hidden_layer_sizes = (200, 200, 200):", accuracy)



# Create and train the MLP classifier with 3 hidden layers
hidden_layer_sizes_a = (1000, 1000, 1000)
mlp_classifier_a = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes_a)
mlp_classifier_a.fit(train_features, train_labels)

# Evaluate the performance of the classifier (a)
accuracy_a = mlp_classifier_a.score(test_features, test_labels)
print("Accuracy for 3 hidden layers hidden_layer_sizes_a = (1000, 1000, 1000):", accuracy_a)



# Create and train the MLP classifier with 5 hidden layers
hidden_layer_sizes_b = (1000, 1000, 1000, 1000, 1000)
mlp_classifier_b = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes_b)
mlp_classifier_b.fit(train_features, train_labels)

# Evaluate the performance of the classifier (b)
accuracy_b = mlp_classifier_b.score(test_features, test_labels)
print("Accuracy for 5 hidden layers: hidden_layer_sizes_b = (1000, 1000, 1000, 1000, 1000)", accuracy_b)


# # fine-tune the classifier on the STL-10 dataset with the following 1%, 10% ,20%,40%,60% of training samples 

# Read the class names from the class_name.txt file
class_names_file = os.path.join(dataset_path, "class_names.txt")
with open(class_names_file, 'r') as f:
    class_names = f.read().splitlines()



# Define a function to evaluate the model performance
def evaluate_model(classifier, train_samples, clstype):
    # Select a subset of training samples based on the given percentage
    train_samples_subset = int(train_samples * len(train_features))
    subset_train_features = train_features[:train_samples_subset]
    subset_train_labels = train_labels[:train_samples_subset]
    
    # Train the classifier on the subset of training samples
    classifier.fit(subset_train_features, subset_train_labels)
    
    # Make predictions on the test set
    predictions = classifier.predict(test_features)
    probabilities = classifier.predict_proba(test_features)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels)
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    
    # Calculate ROC curve and AUC score
    fpr = dict()
    tpr = dict()
    
    roc_auc = dict()
    n_classes = len(class_names)
    for i in range(n_classes):
        if np.sum(test_labels == i) > 0:
            fpr[i], tpr[i], _ = roc_curve(test_labels == i, probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            roc_auc[i] = np.nan
    
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        if not np.isnan(roc_auc[i]):
            plt.plot(fpr[i], tpr[i], label='ROC curve (Class {0})'.format(class_names[i]))
    print(clstype)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    return accuracy, cm, roc_auc

# Define the percentages of training samples to evaluate
percentages = [0.01, 0.1, 0.2, 0.4, 0.6]
#percentages = [0.01]

# Ignore the UndefinedMetricWarning
#warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Evaluate the model performance for each percentage
for percentage in percentages:
    accuracy, cm, roc_auc = evaluate_model(mlp_classifier_a, percentage,"hidden_layer = 3")
    print("For 3 hidden layers - Percentage:", percentage)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("AUC-ROC Scores:")
    for i in range(len(class_names)):
        if not np.isnan(roc_auc[i]):
            print("Class", class_names[i], "-", roc_auc[i])
    print("-----------------------------------")
    
    accuracy, cm, roc_auc = evaluate_model(mlp_classifier_b, percentage,"hidden_layer = 5")
    print("For 5 hidden layers - Percentage:", percentage)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("AUC-ROC Scores:")
    for i in range(len(class_names)):
        if not np.isnan(roc_auc[i]):
            print("Class", class_names[i], "-", roc_auc[i])
    print("-----------------------------------")


