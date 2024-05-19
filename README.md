
   <div style="text-align: center;">
   <a href="https://openreview.net/pdf?id=nqM0lZMevc">
       <img alt="Tests Status" src="https://img.shields.io/badge/OpenReview%20-%20MIDL2024%20-%20green"/>
   </a>
   <a href="https://github.com/mirko-code/MedFed/">
       <img alt="Build Status" src="https://img.shields.io/github/last-commit/mirko-code/MedFed/master?logo=Github"/>
   </a>
   <a href="https://discuss.pytorch.org/t/how-to-install-specific-version-of-torch-2-0-0/177812">
       <img alt="Docs" src="https://img.shields.io/badge/Pytorch-2.0.0+cu117-brightgreen?logo=pytorch&logoColor=red"/>
   </a>
   <a href="https://github.com/mirko-code/MedFed/blob/master/LICENSE">
       <img alt="License" src="https://img.shields.io/github/license/mirko-code/MedFed"/>
   </a>
   </div>


# MedFed

This is the official implementation of our paper ***ASMR: Angular Client Support for Malfunctioning Client Resilience in Federated Learning*** accepted at MIDL 2024.

## Overview
This repository provides a framework for simulating federated learning, specifically designed to study the impact of malfunctioning clients in this context. The framework focuses on medical images, particularly in digital pathology. It considers malfunctioning clients as either malicious or unreliable. Malicious clients deliberately launch untargeted attacks to degrade the global model's performance, while unreliable clients train on suboptimal data containing artifacts. For further information, please refer to our paper.

***Important:*** This framework trains all clients in parallel using multi-threading. Please make sure you have enough resources available.

The following datasets are supported:

| Dataset | Task | Link |
| ------- | ---- | ---- |
| Colectoral Cancer | Classification | https://zenodo.org/records/1214456 |
| Camelyon17 Wilds | Classification | https://wilds.stanford.edu/datasets/#camelyon17 |
| Celeba | Classification | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html |

The following malfunctions are supported:

| Malfunction | Class | Description |
| ----------- | ----- | ----------- |
| Additive-Noise-Attack (ANA) | Malicious | Adds specified amount of noise to the update |
| Sign-Flipping-Attack (SFA) | Malicious | Changes the direction of the vectors of an update |
| Artifacts | Unreliable | Adds pathology-specific artifacts to training data |

To add the artifacts to the training data we used the [FrOoDo](https://github.com/MECLabTUDA/FrOoDo) framework. 

The following protection methods are supported:

| Method | Paper |
| ------ | ----- |
| Multi-Krum | https://proceedings.neurips.cc/paper_files/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf |
| DnC | https://par.nsf.gov/servlets/purl/10286354 |
| Clustered FL | https://iphome.hhi.de/samek/pdf/SatICASSP20.pdf |
| ASMR | https://openreview.net/pdf?id=nqM0lZMevc |


## Getting Started

Installation 
Link to the demos
It is a simulation that uses multi threading

## Demo

1. Download Data
2. Create Config File
3. How to run

## Key Components 

## How to extend
 ### Adding a new aggregation method
 ### Adding new Datasets
 ### Adding a new detection method

 
