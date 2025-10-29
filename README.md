# CXR-Reason

### PhysioNet Access Requirements

The MIMIC-CXR-VQA dataset is constructed from the MIMIC-CXR-JPG (v2.0.0) and Chest ImaGenome (v1.0.0). All these source datasets require a credentialed Physionet license. Due to these requirements and in adherence to the Data Use Agreement (DUA), only credentialed users can access the MIMIC-CXR-VQA dataset files (see Access Policy). To access the source datasets, you must fulfill all of the following requirements:

1. Be a [credentialed user](https://physionet.org/settings/credentialing/)
    - If you do not have a PhysioNet account, register for one [here](https://physionet.org/register/).
    - Follow these [instructions](https://physionet.org/credential-application/) for credentialing on PhysioNet.
    - Complete the "CITI Data or Specimens Only Research" [training course](https://physionet.org/about/citi-course/).
2. Sign the data use agreement (DUA) for each project
    - https://physionet.org/sign-dua/mimic-cxr-jpg/2.0.0/
    - https://physionet.org/sign-dua/chest-imagenome/1.0.0/
    - https://physionet.org/sign-dua/mimiciv/2.2/
3. Download Chest-Imagenome
> wget -r -N -c -np --user {{username}} --ask-password https://physionet.org/files/chest-imagenome/1.0.0/
4. Move processing files and RUN process_data_export.ipynb
```
mv IMAGE_EXCEPTION.txt physionet.org/files/chest-imagenome/1.0.0/
mv CXR-Reason_image_filenames physionet.org/files/chest-imagenome/1.0.0/
mv process_data_export.ipynb physionet.org/files/chest-imagenome/1.0.0/
```
5. Download MIMIC-CXR-JPG files
```
wget -r -N -c -np -nH --cut-dirs=1 --user {{username}} --ask-password -i CXR-Reason_image_filenames --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```
### Example running code (CheXagent)
```
git clone https://github.com/Stanford-AIMI/CheXagent
conda create -n chexagent python=3.10 -y
mv requirements.txt CheXagent
pip install -r requirements.txt
bash run_vqa.sh
```

### Prediction Result Accuracy Computation
Run compute_accuracy.ipynb