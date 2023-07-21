import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import backend as K

image_size = (64, 64)  

# Set the path to the downloaded dataset
dataset_path = 'C://Users//admin//Downloads//VOCtrainval_06-Nov-2007//VOCdevkit//VOC2007'

# Create an empty list to store the preprocessed images
preprocessed_images = []
count = 1
clean_images = []
reconstruction_errors =[]
bottleneck_noise = []
mse_scores = []
mae_scores = []

# Locate the JPEGImages folder within the dataset directory
jpeg_images_path = os.path.join(dataset_path, 'JPEGImages')

# Loop through the images in the JPEGImages folder
for filename in os.listdir(jpeg_images_path):
    # Read the clean image    
    count = count +1
    clean_image = cv2.imread(os.path.join(jpeg_images_path, filename))
    # Resize the clean image
    clean_image = cv2.resize(clean_image, image_size)
    # Normalize the pixel values to a range of 0-1
    clean_image = clean_image.astype('float32') / 255.0
    
    # Add Gaussian noise to the clean image
    noise = np.random.normal(loc=0, scale=0.1, size=clean_image.shape)
    noisy_image = np.clip(clean_image + noise, 0.0, 1.0)  # Clip the pixel values between 0 and 1
    
    # Append the noisy and clean images to the lists
    preprocessed_images.append(noisy_image)
    clean_images.append(clean_image)

# Convert the lists of preprocessed and clean images to numpy arrays
preprocessed_images = np.array(preprocessed_images)
clean_images = np.array(clean_images)


# # Use 80-10-10, train-val-test split

# Perform the train-validation-test split on the preprocessed and clean images
train_images_80_10_10, val_test_images_80_10_10, train_clean_80_10_10, val_test_clean_80_10_10 = train_test_split(preprocessed_images, clean_images, test_size=0.2, random_state=42)
val_images_80_10_10, test_images_80_10_10, val_clean_80_10_10, test_clean_80_10_10 = train_test_split(val_test_images_80_10_10, val_test_clean_80_10_10, test_size=0.5, random_state=42)
# Print the shapes of the train, validation, and test sets
print("Train set shape:", train_images_80_10_10.shape)
print("Validation set shape:", val_images_80_10_10.shape)
print("Test set shape:", test_images_80_10_10.shape)
print("Train clean set shape:", train_clean_80_10_10.shape)
print("Validation clean set shape:", val_clean_80_10_10.shape)
print("Test set clean shape:", test_clean_80_10_10.shape)

# # Use 70-10-20, train-val-test split 

# Perform the train-validation-test split on the preprocessed and clean images
train_images_70_10_20, val_test_images_70_10_20, train_clean_70_10_20, val_test_clean_70_10_20 = train_test_split(preprocessed_images, clean_images, test_size=0.3, random_state=42)
val_images_70_10_20, test_images_70_10_20, val_clean_70_10_20, test_clean_70_10_20 = train_test_split(val_test_images_70_10_20, val_test_clean_70_10_20, test_size=0.67, random_state=42)

# Print the shapes of the train, validation, and test sets
print("Train set shape:", train_images_70_10_20.shape)
print("Validation set shape:", val_images_70_10_20.shape)
print("Test set shape:", test_images_70_10_20.shape)
print("Train clean set shape:", train_clean_70_10_20.shape)
print("Validation clean set shape:", val_clean_70_10_20.shape)
print("Test set clean shape:", test_clean_70_10_20.shape)


# # bottleneck_dim = 256 with 80-10-10

# Set the bottleneck dimension
bottleneck_dim = 256 
# Define the autoencoder model
# Define the input layer
input_img = Input(shape=(image_size[0], image_size[1], 3))
# Encoder
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)
encoded = Dense(bottleneck_dim, activation='relu')(x)
# Decoder
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(3, activation='sigmoid')(x)
# Create the autoencoder model
autoencoder_256_80_10_10 = Model(input_img, decoded)

# Compile the autoencoder model
autoencoder_256_80_10_10.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Print the model summary
autoencoder_256_80_10_10.summary()

# Train the autoencoder model
autoencoder_256_80_10_10.fit(train_images_80_10_10, train_clean_80_10_10,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(val_images_80_10_10, val_clean_80_10_10))

# Evaluate the model on the test set
test_loss = autoencoder_256_80_10_10.evaluate(test_images_80_10_10, test_clean_80_10_10)
print("Test loss:", test_loss)
reconstruction_errors.append(test_loss)
bottleneck_noise.append("autoencoder_256_80_10_10")

denoised_images =autoencoder_256_80_10_10.predict(test_images_80_10_10)
mse_score = np.mean(np.square(test_images_80_10_10 - denoised_images))
mae_score = np.mean(np.abs(test_images_80_10_10 - denoised_images))
print("mse_score",mse_score)
print("mae_score",mae_score)
mse_scores.append(mse_score)
mae_scores.append(mae_score)


# Generate denoised images for a sample of test images
#denoised_images = autoencoder_256_80_10_10.predict(sample_test_images)

# Visualize the original, noisy, and denoised images


fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for i in range(10):
    axes[0, i].imshow(test_clean_80_10_10[i])
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(test_images_80_10_10[i])
    axes[1, i].axis('off')
    axes[1, i].set_title('Noisy')

    axes[2, i].imshow(denoised_images[i])
    axes[2, i].axis('off')
    axes[2, i].set_title('Denoised')

plt.tight_layout()
plt.show()


# 
# # bottleneck_dim = 256 with 70-10-20

# Set the bottleneck dimension
bottleneck_dim = 256 
# Define the autoencoder model
# Define the input layer
input_img = Input(shape=(image_size[0], image_size[1], 3))
# Encoder
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)
encoded = Dense(bottleneck_dim, activation='relu')(x)
# Decoder
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(3, activation='sigmoid')(x)
# Create the autoencoder model
autoencoder_256_70_10_20 = Model(input_img, decoded)

# Compile the autoencoder model
autoencoder_256_70_10_20.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Print the model summary
autoencoder_256_70_10_20.summary()

# Train the autoencoder model
autoencoder_256_70_10_20.fit(train_images_70_10_20, train_clean_70_10_20,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(val_images_70_10_20, val_clean_70_10_20))

# Evaluate the model on the test set
test_loss = autoencoder_256_70_10_20.evaluate(test_images_70_10_20, test_clean_70_10_20)
print("Test loss:", test_loss)
reconstruction_errors.append(test_loss)
bottleneck_noise.append("autoencoder_256_70_10_20")

denoised_images =autoencoder_256_70_10_20.predict(test_images_70_10_20)
mse_score = np.mean(np.square(test_images_70_10_20 - denoised_images))
mae_score = np.mean(np.abs(test_images_70_10_20 - denoised_images))
print("mse_score",mse_score)
print("mae_score",mae_score)
mse_scores.append(mse_score)
mae_scores.append(mae_score)

fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for i in range(10):
    axes[0, i].imshow(test_clean_70_10_20[i])
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(test_images_70_10_20[i])
    axes[1, i].axis('off')
    axes[1, i].set_title('Noisy')

    axes[2, i].imshow(denoised_images[i])
    axes[2, i].axis('off')
    axes[2, i].set_title('Denoised')

plt.tight_layout()
plt.show()


# # bottleneck_dim = 128 with 80-10-10


bottleneck_dim = 128
# Define the autoencoder model

# Define the input layer
input_img = Input(shape=(image_size[0], image_size[1], 3))

# Encoder
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)

# Set the bottleneck dimension
# bottleneck_dim = 128 
encoded = Dense(bottleneck_dim, activation='relu')(x)

# Decoder
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(3, activation='sigmoid')(x)

# Create the autoencoder model
autoencoder_128_80_10_10 = Model(input_img, decoded)

# Compile the autoencoder model
autoencoder_128_80_10_10.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Print the model summary
autoencoder_128_80_10_10.summary()

# Train the autoencoder model
autoencoder_128_80_10_10.fit(train_images_80_10_10, train_clean_80_10_10,
                epochs=10,
                batch_size=40,
                shuffle=True,
                validation_data=(val_images_80_10_10, val_clean_80_10_10))

# Evaluate the model on the test set
test_loss = autoencoder_128_80_10_10.evaluate(test_images_80_10_10, test_clean_80_10_10)
print("Test loss:", test_loss)
reconstruction_errors.append(test_loss)
bottleneck_noise.append("autoencoder_128_80_10_10")

# Generate denoised images for a sample of test images

denoised_images =autoencoder_128_80_10_10.predict(test_images_80_10_10)
mse_score = np.mean(np.square(test_images_80_10_10 - denoised_images))
mae_score = np.mean(np.abs(test_images_80_10_10 - denoised_images))
print("mse_score",mse_score)
print("mae_score",mae_score)
mse_scores.append(mse_score)
mae_scores.append(mae_score)

fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for i in range(10):
    axes[0, i].imshow(test_clean_80_10_10[i])
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(test_images_80_10_10[i])
    axes[1, i].axis('off')
    axes[1, i].set_title('Noisy')

    axes[2, i].imshow(denoised_images[i])
    axes[2, i].axis('off')
    axes[2, i].set_title('Denoised')

plt.tight_layout()
plt.show()


# # bottleneck_dim = 64 with 80-10-10

# Set the bottleneck dimension
bottleneck_dim = 64
# Define the autoencoder model

# Define the input layer
input_img = Input(shape=(image_size[0], image_size[1], 3))

# Encoder
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)

encoded = Dense(bottleneck_dim, activation='relu')(x)

# Decoder
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(3, activation='sigmoid')(x)

# Create the autoencoder model
autoencoder_64_80_10_10 = Model(input_img, decoded)

# Compile the autoencoder model
autoencoder_64_80_10_10.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Print the model summary
autoencoder_64_80_10_10.summary()

# Train the autoencoder model
autoencoder_64_80_10_10.fit(train_images_80_10_10, train_clean_80_10_10,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(val_images_80_10_10, val_clean_80_10_10))

# Evaluate the model on the test set
test_loss = autoencoder_64_80_10_10.evaluate(test_images_80_10_10, test_clean_80_10_10)
print("Test loss:", test_loss)
reconstruction_errors.append(test_loss)
bottleneck_noise.append("autoencoder_64_80_10_10")

# Generate denoised images for a sample of test images

denoised_images =autoencoder_64_80_10_10.predict(test_images_80_10_10)
mse_score = np.mean(np.square(test_images_80_10_10 - denoised_images))
mae_score = np.mean(np.abs(test_images_80_10_10 - denoised_images))
print("mse_score",mse_score)
print("mae_score",mae_score)
mse_scores.append(mse_score)
mae_scores.append(mae_score)

fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for i in range(10):
    axes[0, i].imshow(test_clean_80_10_10[i])
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(test_images_80_10_10[i])
    axes[1, i].axis('off')
    axes[1, i].set_title('Noisy')

    axes[2, i].imshow(denoised_images[i])
    axes[2, i].axis('off')
    axes[2, i].set_title('Denoised')

plt.tight_layout()
plt.show()


# # bottleneck_dim = 32 with 80-10-10

# Set the bottleneck dimension
bottleneck_dim = 32
# Define the autoencoder model

# Define the input layer
input_img = Input(shape=(image_size[0], image_size[1], 3))

# Encoder
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)

encoded = Dense(bottleneck_dim, activation='relu')(x)

# Decoder
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(3, activation='sigmoid')(x)

# Create the autoencoder model
autoencoder_32_80_10_10 = Model(input_img, decoded)

# Compile the autoencoder model
autoencoder_32_80_10_10.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Print the model summary
autoencoder_32_80_10_10.summary()

# Train the autoencoder model
autoencoder_32_80_10_10.fit(train_images_80_10_10, train_clean_80_10_10,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(val_images_80_10_10, val_clean_80_10_10))

# Evaluate the model on the test set
test_loss = autoencoder_32_80_10_10.evaluate(test_images_80_10_10, test_clean_80_10_10)
print("Test loss:", test_loss)

reconstruction_errors.append(test_loss)
bottleneck_noise.append("autoencoder_32_80_10_10")

# Generate denoised images for a sample of test images

denoised_images =autoencoder_32_80_10_10.predict(test_images_80_10_10)
mse_score = np.mean(np.square(test_images_80_10_10 - denoised_images))
mae_score = np.mean(np.abs(test_images_80_10_10 - denoised_images))
print("mse_score",mse_score)
print("mae_score",mae_score)
mse_scores.append(mse_score)
mae_scores.append(mae_score)

# Visualize the original, noisy, and denoised images

fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for i in range(10):
    axes[0, i].imshow(test_clean_80_10_10[i])
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(test_images_80_10_10[i])
    axes[1, i].axis('off')
    axes[1, i].set_title('Noisy')

    axes[2, i].imshow(denoised_images[i])
    axes[2, i].axis('off')
    axes[2, i].set_title('Denoised')

plt.tight_layout()
plt.show()


# # bottleneck_dim = 16 with 80-10-10

# Set the bottleneck dimension
bottleneck_dim = 16
# Define the autoencoder model

# Define the input layer
input_img = Input(shape=(image_size[0], image_size[1], 3))

# Encoder
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)

encoded = Dense(bottleneck_dim, activation='relu')(x)

# Decoder
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(3, activation='sigmoid')(x)

# Create the autoencoder model
autoencoder_16_80_10_10 = Model(input_img, decoded)

# Compile the autoencoder model
autoencoder_16_80_10_10.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Print the model summary
autoencoder_16_80_10_10.summary()

# Train the autoencoder model
autoencoder_16_80_10_10.fit(train_images_80_10_10, train_clean_80_10_10,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(val_images_80_10_10, val_clean_80_10_10))

# Evaluate the model on the test set
test_loss = autoencoder_16_80_10_10.evaluate(test_images_80_10_10, test_clean_80_10_10)
print("Test loss:", test_loss)
reconstruction_errors.append(test_loss)
bottleneck_noise.append("autoencoder_16_80_10_10")

# Generate denoised images for a sample of test images

denoised_images =autoencoder_16_80_10_10.predict(test_images_80_10_10)
mse_score = np.mean(np.square(test_images_80_10_10 - denoised_images))
mae_score = np.mean(np.abs(test_images_80_10_10 - denoised_images))
print("mse_score",mse_score)
print("mae_score",mae_score)
mse_scores.append(mse_score)
mae_scores.append(mae_score)

# Visualize the original, noisy, and denoised images
fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for i in range(10):
    axes[0, i].imshow(test_clean_80_10_10[i])
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(test_images_80_10_10[i])
    axes[1, i].axis('off')
    axes[1, i].set_title('Noisy')

    axes[2, i].imshow(denoised_images[i])
    axes[2, i].axis('off')
    axes[2, i].set_title('Denoised')

plt.tight_layout()
plt.show()

for i in range(len(bottleneck_noise)):
    print("Model ",bottleneck_noise[i])
    print("Loss",reconstruction_errors[i])
    print("MSE Error",mse_scores[i])
    print("MAE Error",mae_scores[i])


# # Choose the bottleneck dimension 256, and re-run the autoencoder using masking strategy: (mask the following 20% of pixels in the image, i.e., set the pixel value to (0,0,0))


bottleneck_dim = 256
preprocessed_images = []
clean_images = []

# Loop through the images in the JPEGImages folder
for filename in os.listdir(jpeg_images_path):
    # Read the clean image
    clean_image = cv2.imread(os.path.join(jpeg_images_path, filename))
    # Resize the clean image
    clean_image = cv2.resize(clean_image, image_size)
    # Normalize the pixel values to a range of 0-1
    clean_image = clean_image.astype('float32') / 255.0
    
    # Apply masking to the clean image
    mask = np.random.choice([0, 1], size=clean_image.shape[:2], p=[0.2, 0.8])
    masked_image = clean_image * np.expand_dims(mask, axis=-1)
    
    # Append the masked and clean images to the lists
    preprocessed_images.append(masked_image)
    clean_images.append(clean_image)

# Convert the lists of preprocessed and clean images to numpy arrays
preprocessed_images = np.array(preprocessed_images)
clean_images = np.array(clean_images)

# Perform the train-validation-test split on the preprocessed and clean images
train_images, val_test_images, train_clean, val_test_clean = train_test_split(preprocessed_images, clean_images, test_size=0.2, random_state=42)
val_images, test_images, val_clean, test_clean = train_test_split(val_test_images, val_test_clean, test_size=0.5, random_state=42)

# Define the custom loss function with masking
def masked_mse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0.0), K.floatx())
    masked_squared_error = K.square(mask * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error) / K.sum(mask)
    return masked_mse

# Define the autoencoder model
input_img = Input(shape=(image_size[0], image_size[1], 3))
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)
encoded = Dense(bottleneck_dim, activation='relu')(x)
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(3, activation='sigmoid')(x)

autoencoder_20per_Masking = Model(input_img, decoded)

# Compile the autoencoder model with the custom loss function
autoencoder_20per_Masking.compile(optimizer=Adam(learning_rate=0.001), loss=masked_mse)

# Print the model summary
autoencoder_20per_Masking.summary()

# Train the autoencoder model
autoencoder_20per_Masking.fit(train_images, train_clean,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(val_images, val_clean))

# Evaluate the model on the test set
test_loss = autoencoder_20per_Masking.evaluate(test_images, test_clean)
print("Test loss:", test_loss)
reconstruction_errors.append(test_loss)
bottleneck_noise.append("autoencoder_20per_Masking")

# Generate denoised images for a sample of test images


denoised_images =autoencoder_20per_Masking.predict(test_images)
mse_score = np.mean(np.square(test_images - denoised_images))
mae_score = np.mean(np.abs(test_images - denoised_images))
print("mse_score",mse_score)
print("mae_score",mae_score)
mse_scores.append(mse_score)
mae_scores.append(mae_score)

# Visualize the original, masked, and denoised images
fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for i in range(10):
    axes[0, i].imshow(test_clean[i])
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(test_images[i])
    axes[1, i].axis('off')
    axes[1, i].set_title('Masked')

    axes[2, i].imshow(denoised_images[i])
    axes[2, i].axis('off')
    axes[2, i].set_title('Denoised')

plt.tight_layout()
plt.show()


# ## Choose the bottleneck dimension 256, and re-run the autoencoder using masking strategy: (mask the following 40% of pixels in the image, i.e., set the pixel value to (0,0,0))

# Create an empty list to store the preprocessed images
preprocessed_images = []
clean_images = []
bottleneck_dim = 256

# Loop through the images in the JPEGImages folder
for filename in os.listdir(jpeg_images_path):
    # Read the clean image
    clean_image = cv2.imread(os.path.join(jpeg_images_path, filename))
    # Resize the clean image
    clean_image = cv2.resize(clean_image, image_size)
    # Normalize the pixel values to a range of 0-1
    clean_image = clean_image.astype('float32') / 255.0
    
    # Apply masking to the clean image
    mask = np.random.choice([0, 1], size=clean_image.shape[:2], p=[0.4, 0.6])
    masked_image = clean_image * np.expand_dims(mask, axis=-1)
    
    # Append the masked and clean images to the lists
    preprocessed_images.append(masked_image)
    clean_images.append(clean_image)

# Convert the lists of preprocessed and clean images to numpy arrays
preprocessed_images = np.array(preprocessed_images)
clean_images = np.array(clean_images)

# Perform the train-validation-test split on the preprocessed and clean images
train_images, val_test_images, train_clean, val_test_clean = train_test_split(preprocessed_images, clean_images, test_size=0.2, random_state=42)
val_images, test_images, val_clean, test_clean = train_test_split(val_test_images, val_test_clean, test_size=0.5, random_state=42)

# Define the custom loss function with masking
def masked_mse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0.0), K.floatx())
    masked_squared_error = K.square(mask * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error) / K.sum(mask)
    return masked_mse

# Define the autoencoder model
input_img = Input(shape=(image_size[0], image_size[1], 3))
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)
encoded = Dense(bottleneck_dim, activation='relu')(x)
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(3, activation='sigmoid')(x)

autoencoder_40per_masking = Model(input_img, decoded)

# Compile the autoencoder model with the custom loss function
autoencoder_40per_masking.compile(optimizer=Adam(learning_rate=0.001), loss=masked_mse)

# Print the model summary
autoencoder_40per_masking.summary()

# Train the autoencoder model
autoencoder_40per_masking.fit(train_images, train_clean,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(val_images, val_clean))

# Evaluate the model on the test set
test_loss = autoencoder_40per_masking.evaluate(test_images, test_clean)
print("Test loss:", test_loss)
reconstruction_errors.append(test_loss)
bottleneck_noise.append("autoencoder_40per_Masking")
# Generate denoised images for a sample of test images
#denoised_images = autoencoder_40per_masking.predict(test_images[:10])

denoised_images =autoencoder_40per_masking.predict(test_images)
mse_score = np.mean(np.square(test_images - denoised_images))
mae_score = np.mean(np.abs(test_images - denoised_images))
print("mse_score",mse_score)
print("mae_score",mae_score)
mse_scores.append(mse_score)
mae_scores.append(mae_score)


# Visualize the original, masked, and denoised images
fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for i in range(10):
    axes[0, i].imshow(test_clean[i])
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(test_images[i])
    axes[1, i].axis('off')
    axes[1, i].set_title('Masked')

    axes[2, i].imshow(denoised_images[i])
    axes[2, i].axis('off')
    axes[2, i].set_title('Denoised')

plt.tight_layout()
plt.show()


# ## Choose the bottleneck dimension 256, and re-run the autoencoder using masking strategy: (mask the following 60% of pixels in the image, i.e., set the pixel value to (0,0,0))

# Create an empty list to store the preprocessed images
preprocessed_images = []
clean_images = []
bottleneck_dim = 256

# Locate the JPEGImages folder within the dataset directory
jpeg_images_path = os.path.join(dataset_path, 'JPEGImages')

# Loop through the images in the JPEGImages folder
for filename in os.listdir(jpeg_images_path):
    # Read the clean image
    clean_image = cv2.imread(os.path.join(jpeg_images_path, filename))
    # Resize the clean image
    clean_image = cv2.resize(clean_image, image_size)
    # Normalize the pixel values to a range of 0-1
    clean_image = clean_image.astype('float32') / 255.0
    
    # Apply masking to the clean image
    mask = np.random.choice([0, 1], size=clean_image.shape[:2], p=[0.6, 0.4])
    masked_image = clean_image * np.expand_dims(mask, axis=-1)
    
    # Append the masked and clean images to the lists
    preprocessed_images.append(masked_image)
    clean_images.append(clean_image)

# Convert the lists of preprocessed and clean images to numpy arrays
preprocessed_images = np.array(preprocessed_images)
clean_images = np.array(clean_images)

# Perform the train-validation-test split on the preprocessed and clean images
train_images, val_test_images, train_clean, val_test_clean = train_test_split(preprocessed_images, clean_images, test_size=0.2, random_state=42)
val_images, test_images, val_clean, test_clean = train_test_split(val_test_images, val_test_clean, test_size=0.5, random_state=42)

# Define the custom loss function with masking
def masked_mse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0.0), K.floatx())
    masked_squared_error = K.square(mask * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error) / K.sum(mask)
    return masked_mse

# Define the autoencoder model
input_img = Input(shape=(image_size[0], image_size[1], 3))
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)
encoded = Dense(bottleneck_dim, activation='relu')(x)
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(3, activation='sigmoid')(x)

autoencoder_60per_masking = Model(input_img, decoded)

# Compile the autoencoder model with the custom loss function
autoencoder_60per_masking.compile(optimizer=Adam(learning_rate=0.001), loss=masked_mse)

# Print the model summary
autoencoder_60per_masking.summary()

# Train the autoencoder model
autoencoder_60per_masking.fit(train_images, train_clean,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(val_images, val_clean))

# Evaluate the model on the test set
test_loss = autoencoder_60per_masking.evaluate(test_images, test_clean)
print("Test loss:", test_loss)
reconstruction_errors.append(test_loss)
bottleneck_noise.append("autoencoder_60per_Masking")
# Generate denoised images for a sample of test images

denoised_images =autoencoder_60per_masking.predict(test_images)
mse_score = np.mean(np.square(test_images - denoised_images))
mae_score = np.mean(np.abs(test_images - denoised_images))
print("mse_score",mse_score)
print("mae_score",mae_score)
mse_scores.append(mse_score)
mae_scores.append(mae_score)


# Visualize the original, masked, and denoised images

fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for i in range(10):
    axes[0, i].imshow(test_clean[i])
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(test_images[i])
    axes[1, i].axis('off')
    axes[1, i].set_title('Masked')

    axes[2, i].imshow(denoised_images[i])
    axes[2, i].axis('off')
    axes[2, i].set_title('Denoised')

plt.tight_layout()
plt.show()


# ## Choose the bottleneck dimension 256, and re-run the autoencoder using masking strategy: (mask the following 80% of pixels in the image, i.e., set the pixel value to (0,0,0))


# Create an empty list to store the preprocessed images
preprocessed_images = []
clean_images = []
bottleneck_dim = 256

# Loop through the images in the JPEGImages folder
for filename in os.listdir(jpeg_images_path):
    # Read the clean image
    clean_image = cv2.imread(os.path.join(jpeg_images_path, filename))
    # Resize the clean image
    clean_image = cv2.resize(clean_image, image_size)
    # Normalize the pixel values to a range of 0-1
    clean_image = clean_image.astype('float32') / 255.0
    
    # Apply masking to the clean image
    mask = np.random.choice([0, 1], size=clean_image.shape[:2], p=[0.2, 0.8])
    masked_image = clean_image * np.expand_dims(mask, axis=-1)
    
    # Append the masked and clean images to the lists
    preprocessed_images.append(masked_image)
    clean_images.append(clean_image)

# Convert the lists of preprocessed and clean images to numpy arrays
preprocessed_images = np.array(preprocessed_images)
clean_images = np.array(clean_images)

# Perform the train-validation-test split on the preprocessed and clean images
train_images, val_test_images, train_clean, val_test_clean = train_test_split(preprocessed_images, clean_images, test_size=0.2, random_state=42)
val_images, test_images, val_clean, test_clean = train_test_split(val_test_images, val_test_clean, test_size=0.5, random_state=42)

# Define the custom loss function with masking
def masked_mse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0.0), K.floatx())
    masked_squared_error = K.square(mask * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error) / K.sum(mask)
    return masked_mse

# Define the autoencoder model
input_img = Input(shape=(image_size[0], image_size[1], 3))
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)
encoded = Dense(bottleneck_dim, activation='relu')(x)
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(3, activation='sigmoid')(x)

autoencoder_80per_masking = Model(input_img, decoded)

# Compile the autoencoder model with the custom loss function
autoencoder_80per_masking.compile(optimizer=Adam(learning_rate=0.001), loss=masked_mse)

# Print the model summary
autoencoder_80per_masking.summary()

# Train the autoencoder model
autoencoder_80per_masking.fit(train_images, train_clean,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(val_images, val_clean))

# Evaluate the model on the test set
test_loss = autoencoder_80per_masking.evaluate(test_images, test_clean)
print("Test loss:", test_loss)
reconstruction_errors.append(test_loss)
bottleneck_noise.append("autoencoder_80per_Masking")
# Generate denoised images for a sample of test images

denoised_images =autoencoder_80per_masking.predict(test_images)

mse_score = np.mean(np.square(test_images - denoised_images))
mae_score = np.mean(np.abs(test_images - denoised_images))
print("mse_score",mse_score)
print("mae_score",mae_score)
mse_scores.append(mse_score)
mae_scores.append(mae_score)

# Visualize the original, masked, and denoised images
fig, axes = plt.subplots(3, 10, figsize=(20, 6))
for i in range(10):
    axes[0, i].imshow(test_clean[i])
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(test_images[i])
    axes[1, i].axis('off')
    axes[1, i].set_title('Masked')

    axes[2, i].imshow(denoised_images[i])
    axes[2, i].axis('off')
    axes[2, i].set_title('Denoised')

plt.tight_layout()
plt.show()


# ## Plot reconstruction error for every autoencoder model


for i in range(len(bottleneck_noise)):
    print("Model ",bottleneck_noise[i],"=======","Loss" ,reconstruction_errors[i]  )
    #print("Loss",reconstruction_errors[i])
    print("MSE Error",mse_scores[i])
    print("MAE Error",mae_scores[i])




# Plot the reconstruction errors
plt.plot(bottleneck_noise, reconstruction_errors, marker='o')
plt.xlabel('Bottleneck Dimension')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs Bottleneck Dimension')
ax= plt.subplot()
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.show()

# Plot the MSE and MAE scores
bottleneck_dims = bottleneck_noise 
plt.plot(bottleneck_noise, mse_scores, marker='o', label='MSE')
plt.plot(bottleneck_noise, mae_scores, marker='o', label='MAE')
plt.xlabel('Bottleneck Dimension')
plt.ylabel('Score')
plt.title('Evaluation Metrics')
ax= plt.subplot()
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.legend()
plt.show()


# # Use any other metric of your choice (apart from MSE, MAE) to judge the image quality

# # Structural Content (SC)

def calculate_sc(original_image, reconstructed_image):
    # Convert images to grayscale if they are in color
    if len(original_image.shape) == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if len(reconstructed_image.shape) == 3:
        reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the absolute difference between original and reconstructed images
    abs_diff = np.abs(original_image - reconstructed_image)
    
    # Compute the structural similarity by taking the mean of the absolute difference
    sc = np.mean(abs_diff)
    
    return sc

# Example usage
sc_score = calculate_sc(test_images, denoised_images)
print("SC_Score",sc_score)


# # save final model


# Set the desired image size
image_size = (64, 64)

# Set the path to the downloaded dataset
dataset_path = 'C://Users//admin//Downloads//VOCtrainval_06-Nov-2007//VOCdevkit//VOC2007'

# Create an empty list to store the preprocessed images
preprocessed_images = []
clean_images = []
bottleneck_dim = 256

# Locate the JPEGImages folder within the dataset directory
jpeg_images_path = os.path.join(dataset_path, 'JPEGImages')

# Loop through the images in the JPEGImages folder
for filename in os.listdir(jpeg_images_path):
    # Read the clean image
    clean_image = cv2.imread(os.path.join(jpeg_images_path, filename))
    # Resize the clean image
    clean_image = cv2.resize(clean_image, image_size)
    # Normalize the pixel values to a range of 0-1
    clean_image = clean_image.astype('float32') / 255.0
    
    # Apply masking to the clean image
    mask = np.random.choice([0, 1], size=clean_image.shape[:2], p=[0.2, 0.8])
    masked_image = clean_image * np.expand_dims(mask, axis=-1)
    
    # Append the masked and clean images to the lists
    preprocessed_images.append(masked_image)
    clean_images.append(clean_image)

# Convert the lists of preprocessed and clean images to numpy arrays
print("image readng complete")
preprocessed_images = np.array(preprocessed_images)
clean_images = np.array(clean_images)
print("np array complete")

# Perform the train-validation-test split on the preprocessed and clean images
train_images, val_test_images, train_clean, val_test_clean = train_test_split(preprocessed_images, clean_images, test_size=0.2, random_state=42)
val_images, test_images, val_clean, test_clean = train_test_split(val_test_images, val_test_clean, test_size=0.5, random_state=42)

# Define the custom loss function with masking
def masked_mse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0.0), K.floatx())
    masked_squared_error = K.square(mask * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error) / K.sum(mask)
    return masked_mse

# Define the autoencoder model
input_img = Input(shape=(image_size[0], image_size[1], 3))
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)
encoded = Dense(bottleneck_dim, activation='relu')(x)
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(3, activation='sigmoid')(x)

autoencoder_BD256_20msk = Model(input_img, decoded)

# Compile the autoencoder model with the custom loss function
autoencoder_BD256_20msk.compile(optimizer=Adam(learning_rate=0.001), loss=masked_mse)

# Print the model summary
autoencoder_BD256_20msk.summary()

# Train the autoencoder model
autoencoder_BD256_20msk.fit(train_images, train_clean,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(val_images, val_clean))

# Evaluate the model on the test set
test_loss = autoencoder_BD256_20msk.evaluate(test_images, test_clean)
print("Test loss:", test_loss)


denoised_images =autoencoder_BD256_20msk.predict(test_images)

mse_score = np.mean(np.square(test_images - denoised_images))
mae_score = np.mean(np.abs(test_images - denoised_images))
print("mse_score",mse_score)
print("mae_score",mae_score)

# Save the autoencoder model
autoencoder_BD256_20msk.save("autoencoder_BD256_20msk_model.h5")


