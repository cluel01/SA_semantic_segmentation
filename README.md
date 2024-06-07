# Project Overview
Climate change and anthropogenic pressure are driving ecosystems toward tipping points, where their structure and functioning could be suddenly or irreversibly altered. This project leverages very high-resolution images and deep learning methods to map over 4 billion individual trees and shrubs across all ecosystem types in South Africa. Our findings reveal significant discrepancies in tree and shrub coverage compared to existing maps, suggesting potential shifts in ecosystem structure and functioning.

# Key Findings
Mapped approximately 4,159,227,100 individual trees and shrubs in South Africa.
Discovered 151.15% larger tree coverage in shrublands compared to previous maps.
Detected trees in areas considered shrublands and forests, indicating a potential shift in ecosystem definitions.
Our results allow nationwide localization and quantification of ecological phenomena such as forest transition, bush encroachment, and woody plant invasion.
# Methodology
## Data Collection
High-resolution Satellite Imagery: Used imagery from 2009 to 2020 with resolutions of 40 cm per pixel (2009-2016) and 20 cm per pixel (2017-2020).
Training Data: Combined labeled data from Rwanda and manually labeled regions from South Africa.
## Model Training
Model: UNet and SegFormer models for segmenting individual tree canopies.
Initial Training: Used 100,000 labeled trees from Rwanda (2008, 20 cm resolution) and 50,000 labeled trees from Rwanda (2019, upscaled to 20 cm resolution).
Additional Training: Included 40,000 manually labeled trees from South Africa with varying resolutions (20-40 cm).
Data Augmentation: Applied extensive techniques to make the model resilient to varying resolutions and color intensities.
Validation Metric: Used mean Intersection over Union (mIoU) for validation.
