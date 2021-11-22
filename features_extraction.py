# Import libraries
import os
import pandas as pd
import numpy as np
from skimage import io
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import resize
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import kurtosis, skew, entropy


# Get an image and returns first and second order features of the pixels array 
def get_image_features(image_path):
    # Load the image and converts it into a 2-dimensional numpy array with the pixels values
    image = io.imread(image_path, as_gray = True) * 255
    # Resize image to 512 x 512
    image = resize(image, (512, 512))
    # Converts the float values of the 2-dimensional pixels array into uint8
    image = image.astype(np.uint8)
    
    # Calculate the mean, variance and standard deviation of the 2-dimensional pixels array with numpy functions
    mean = np.mean(image)
    variance = np.var(image)
    std = np.std(image)
    
    # Converts the 2-dimensional pixels array into 1-dimensional
    image_1da = image.flatten()
    
    # Calculate the skewness, kurtosis and entropy of the 1-dimensional array with scipy.stats functions
    skewness = skew(image_1da)
    kurtos = kurtosis(image_1da)
    entro = entropy(image_1da)

    # Calculate the grey-level-co-ocurrence matrix with skimage functions
    # The pixel pair distance offset used is 1
    # The pixel pair angles used are 0, pi/4, pi/2 and 3pi/4
    GLCM = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

    # Calculate texture properties of the grey-level-co-ocurrence matrix 
    contrast = greycoprops(GLCM, 'contrast')[0, 0]
    dissimilarity = greycoprops(GLCM, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(GLCM, 'homogeneity')[0, 0]
    asm = greycoprops(GLCM, 'ASM')[0, 0]
    energy = greycoprops(GLCM, 'energy')[0, 0]
    correlation = greycoprops(GLCM, 'correlation')[0, 0]
    
    # Returns all the features values of the image
    return mean, variance, std, skewness, kurtos, entro, contrast, dissimilarity, homogeneity, asm, energy, correlation


# Gets the categorical value and assign it to a numeric value between 0 and 3
def label_class(row):
    if row['label_name'] == 'no_tumor':
        return 0
    elif row['label_name'] == 'glioma_tumor':
        return 1
    elif row['label_name'] == 'meningioma_tumor':
        return 2
    else:
        return 3


# Images directory
images_dir = './data/Training'

# Create a dataframe to store the values of the features of each image
df = pd.DataFrame(columns = ('image_name', 'mean', 'variance', 'std', 'skewness', 'kurtosis', 'entropy','contrast', 'dissimilarity', 'homogeneity', 'asm', 'energy', 'correlation', 'label_name'))

# A variable to iterate with
image_num = 1

# For loops to iterate between each image of the folders
for classes in os.listdir(images_dir):
    for images in os.listdir(images_dir + '/' + classes):
        # Image path
        inputfile = images_dir + '/' + classes + '/' + images
        
        # Call features_extraction function and save the features values in a variable
        features = get_image_features(inputfile)
        
        # Assign each feature value into its variable
        (mean, variance, std, skewness, kurtos, entro,contrast, dissimilarity, homogeneity, asm, energy, correlation) = features
        
        # Add each image feature value into the dataframe
        df.loc[image_num] = images, mean, variance, std, skewness, kurtos, entro, contrast, dissimilarity, homogeneity, asm, energy, correlation, classes
        
        image_num += 1

# Apply the label_class function into each row of the dataframe      
df['label'] = df.apply(lambda row: label_class(row), axis = 1)

# Shuffle the rows in the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Return a csv file with the dataframe
df.to_csv('data/brain_tumor_dataset.csv')

print('Ready')