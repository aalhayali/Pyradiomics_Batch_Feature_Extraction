# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as sns
import os
import six
import radiomics
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import SimpleITK as sitk

# Radiomics Set-up
params = os.path.join(os.getcwd(), '/Params.yaml') # Assign a parameters file
extractor = featureextractor.RadiomicsFeatureExtractor(params)

# Initialize features dictionary
features = {}

# Loop through all the data samples
for case_id in range(1, 214):
    path = 'C:\My files\Research\Rectal Gas Localizer DICOM + Segmentations\\full\images\\'.format(case_id)
    path2 = 'C:\My files\Research\Rectal Gas Localizer DICOM + Segmentations\\full\GT\\'.format(case_id)
    image = sitk.ReadImage(path + "M{}.nii".format(case_id))
    mask = sitk.ReadImage(path2 + "M{}SEG.nii".format(case_id))
    features[case_id] = extractor.execute(image, mask) # Assign the image extracted features to the dictionary

feature_names = list(sorted(filter(lambda k: k.startswith("original_"), features[1]))) #Keep features that start with original (all other ones are noise from the input)

# Make a numpy array of all the features
samples = np.zeros((213, len(feature_names)))
for case_id in range(1,214):
    a = np.array([])
    for feature_name in feature_names:
        a = np.append(a, features[case_id][feature_name])
    samples[case_id - 1, :] = a
    samples = np.nan_to_num(samples)

# Construct a pandas dataframe from the samples
df_sfn = pd.DataFrame(data=samples, columns=feature_names)

# Draw a features heatmap
corr = df_sfn.corr()
sns.heatmap(corr, vmax=.8, square=True)