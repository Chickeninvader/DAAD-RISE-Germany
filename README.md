# Traffic Accident Classification Project

## Overview
This project implements a binary classification model for predicting traffic accidents from video data, achieving a 0.75 F1 score. The model can be used to automatically collect and classify traffic accident data, particularly focusing on ego-vehicle (first-person perspective) scenarios.

## Features
- Video-based traffic accident classification
- Support for multiple deep learning architectures
- Visualization tools for model predictions

## Models
The project implements two main architectures:
1. YOLOv1 with LSTM
   - Uses pretrained weights from traffic datasets
   - Variants with and without fully connected layers
   
2. Video Swin Transformer (Swin3D)
   - Supports both pretrained (Kinetics 400 dataset) and from-scratch training
   - Best performing model with 0.756 F1 score when using pretrained weights

## Dataset
The project uses three datasets:
- Dashcam Video Dataset: 50 videos with manual critical scenario labels
- Car Crash Dataset: 800 videos (5 seconds each) https://github.com/Cogito2012/CarCrashDataset
- BDD100K Dataset: 1000 videos (40 seconds each) http://bdd-data.berkeley.edu/download.html

### Training
```bash
python critical_classification/training_torch.py \
 --model_name Swin3D \
 --image_batch_size 10 \
 --additional_config train_from_scratch

```

### Inference
```bash
python critical_classification/inference.py \
  --model_name Swin3D \
  --image_batch_size 10 \
  --pretrained_path [path_to_model] \
  --infer_all_video
```

Model Configuration Options
For YOLOv1_video:
```bash
--additional_config no_fc
```
For Swin3D:
```bash
--additional_config train_from_scratch
```
Experimental Setup

## Results

| Model                        | F1 Score (best) |  
|------------------------------|-----------------|  
| Swin3D                       | 0.178           |  
| YOLOv1 + LSTM                | 0.600           |  
| **Swin3D with Pretrained**   | **0.756**       |  
| YOLOv1 + LSTM (No FC Layer)  | 0.601           |  

The project includes SLURM scripts for HPC environments:

GPU: A100, Time allocation: 1 day

For more detail please refer to the report

## Future Work

Implementation of object tracking in bird's-eye-view format
Automated annotation system development
Expansion of the traffic accident dataset
Enhancement of safety analysis capabilities for autonomous vehicles

## Acknowledgement

This project is funded by the DAAD (Deutscher Akademischer Austauschdienst) Research Internships in Science & Engineering (RISE).  

The project is conducted at the Institute for Technologies and Management of Digital Transformation at the University of Wuppertal (Bergische Universität Wuppertal). I am supervised by Adwait Chandorkar, a Ph.D. student specializing in machine learning and computer vision, under the guidance of Prof. Dr.-Ing. Tobias Meisen. Their expertise and mentorship have been instrumental in shaping this project.  
