# Generating Darker Skin Tone for Skin Lesion Using Deep Convolutional Generative Adversarial Network (DCGAN) 

In an effort to foster research in the automated diagnosis of skin conditions, Tschandl et al. introduced the “Human Against Machine with 10000 training images” (HAM10000) dataset in 2018. This valuable resource comprises 10,015 dermatoscopic images that were already annotated. While the HAM10000 dataset offers a large number of high-quality labeled images, its capacity to represent the broader population is limited by a heavy skew toward lighter skin tones, primarily due to the demography of the data collection site (Austria). Specifically, based on the Fitzpatrick skin type scale, the dataset predominantly includes samples corresponding to type I and II. This bias raises concerns that models trained on this dataset may poorly perform when applied to individuals with darker skin tones.  

To address this critical limitation, our research project proposes to use DCGAN to generate synthetic skin lesion images with darker skin tones, aligning with Fitzpatrick scale classification III through VI. By comparing the ResNet 18 classifier performance between models trained on the original HAM10000 dataset and those trained on an augmented dataset incorporating our synthetic images, we aim to evaluate the reliability and potential benefits of using synthetic data to mitigate the issue of skin tone bias in dermatological image analysis. 

Reference:  

 Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific data, 5, 180161. https://doi.org/10.1038/sdata.2018.161 

 

## Project Structure (Initial) 

`project_root/`  
`├── main.py`: Entry point for downloading data, training, and testing the model  
`└── requirements.txt`: Required Python packages  

## Project Structure (After all possible executions) 

`project_root/`  
`├── data/`: Stores all datasets and saved model checkpoints  
`|   ├── fresh_data/`: COriginal fitzpatrick17k CSV, filtered to a given label threshhold and images downloaded  
`|   |   ├── src_images/`: The source images for fitzpatrick17k, grouped into subdirs by their label  
`|   |   |   └── subdirs/`  
`|   |   ├── fitzpatrick17k.csv`: Original fitzpatrick17k csv  
`|   |   ├── unsaved_rows.csv`: The rows which failed to download, often due to broken links  
`|   |   └── downloaded_fitzpatrick17k.csv`: Filtered CSV containing the skin lesion entries of interest, with the addition of their local paths  
`|   ├── prepared_data/`: Contains the pre-filtered fitzpatrick17k data and model checkpoints downloaded from our GitHub  
`|   |   ├── src_images/`: The source images for fitzpatrick17k, grouped into subdirs by their label  
`|   |   |   └── subdirs/`  
`|   |   └── downloaded_fitzpatrick17k.csv`: Filtered CSV containing the skin lesion entries of interest, with the addition of their local paths  
`|   └── HAM10000`:  
`|       ├── HAM10000_images_part_1/`: Subdirectory containing part of the images from the HAM10000 dataset  
`|       ├── HAM10000_images_part_2/`: Subdirectory containing the remaining images from the HAM10000 dataset  
`|       ├── HAM10000_metadata.csv`: Contains labels and IDs used to map HAM10000 entries to fitzpatrick17k  
`|       └── *.csv`: Various other packaged CSV files from the dataset that were not used in this project  
`├── results/`: Stores plots, outputs, and locally generate model checkpoints.  
`|   ├── generated/`: Synthetic GAN generated images for augmentation of datasets to be used by the classifier  
`|   └── *.*`: Saved output metrics, visualizations, and model checkpoints  
`├── main.py`: Entry point for downloading data, training, and testing the model  
`├── requirements.txt`: Entry point for downloading data, training, and testing the model  
`├── README.md`: The current file describing project layout and instructions  
`└── kaggle.json`: Contains API key for use in downloading HAM10000  

## How to Run 

From your terminal, navigate to your project directory and first install any dependencies using:

`pip install -r requirements.txt`

or an equivalent command. Then run according to your desired behavior, run:

`py main.py --train` or `py main.py --test`

Note: `py` may be `python`, `python3`, or another alias depending on your local setup.

 

### Options 

`--fresh [int]`: Downloads the Fitzpatrick17k dataset and filters to N samples per class (recommended: 3–10). If omitted, it downloads a pre-filtered version from GitHub. 

`--redownload`: Deletes any locally stored data and redownloads from the chosen source (raw or pre-filtered). 

`--train`: Trains the model

`--traingan`: Trains the GAN portion, while only testing the ResNet18 classifier

`--trainclf`: Trains the ResNet18 classifier, while only testing the GAN portion

`--test`: Tests all models using pretrained checkpoints downloaded from GitHub. 

`--testlocal`: Tests the model using locally trained checkpoints. 

`--use_synth`: Add generated images to the original HAM10000 dataset for classifier training. (Default=True, redundant flag)

`--use_baseline`: Do not add generated images to the original HAM10000 dataset for classifier training. (Disables use_synth) 

`--show`: Displays plots after training/testing. If not set, plots are saved to the results/ directory.

## Dataset Source 

The Fitzpatrick17k dataset includes labeled images categorized into six Fitzpatrick skin types. Images have been resized and standardized to 64x64 3-channel RGB inputs. 

The HAM10000 (Human Against Machine with 10000 training images) includes 10,015 high-resolution dermatoscopic images categorized into seven skin lesion types. Images have been resized and standardized to 150x150 3-channel RGB inputs. 

 

## Notes 

Ensure all dependencies are installed (`requirements.txt`). 

CUDA is optional but recommended. The model can run on CPU, but training will be slower. With CUDA, the whole DCGAN and ResNet18 training will take up to an hour.  

Results include generated samples and visualizations of model performances. 