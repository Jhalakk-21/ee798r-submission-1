# Consistent MS/HS Fusion Net 
For multispectral and hyperspectral image fusion in the case that spectral and spatial responses of training and testing data are consistent


Outline:

    Folder structure
    Usage
    Citation
    
Folder structure:

    rowData\    : Original multispectral images (MSI) data 
        |-- CAVEdata\      : CAVE data set
        |       |-- complete_ms_data\       : The images of data
        |       |-- response coefficient.mat: A matrix of exploited response coefficient
    CAVE_dataReader.py     : The data reading and preparing code
    CAVEmain.py            : Main code for training and testing 
    MHFnet.py              : Code of MHF-net 
    MyLib.py               : Some used code
    TestSample.mat         : A example testing data

Usage:

Create a conda environment using the command and install the requirements.
conda create --name mhf-net python=3.10
conda activate mhf-net
pip install -r requirements.txt

The command to run the CAVEmain.py file
python CAVEmain.py --mode = train --prepare = Yes

The flag prepare is to prepare the data.
To train and test on CAVE data set, you must first download the CAVE data set form http://www.cs.columbia.edu/CAVE/databases/multispectral/, and put the data in the folder ./ rowData/CAVEdata/complete_ms_data/ just like:

The paths used in the code are absolute and hence needs to be changed from system to system.


Citation:

    Qi Xie, Minghao Zhou, Qian Zhao, Deyu Meng*, Wangmeng Zuo and Zongben Xu
    Multispectral and Hyperspectral Image Fusion by MS/HS Fusion Net[C]
    2019 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, 2019.

    Qi Xie, Minghao Zhou, Qian Zhao, Zongben Xu and Deyu Meng* 
    MHF-Net: An Interpretable Deep Network for Multispectral and Hyperspectral Image Fusion
    IEEE transactions on pattern analysis and machine intelligence, 2020.

