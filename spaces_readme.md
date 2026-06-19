---
title: OralGuard
emoji: 🦷
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
license: mit
---

# OralGuard — Uncertainty-Aware Dental Pathology Detection

A research demonstration of the OralGuard pipeline for detecting dental pathologies on panoramic X-rays.

## Features
- **Tooth Detection**: YOLOv8m trained on DENTEX 2023 (mAP@50: 0.548)
- **FDI Notation**: Automatic ISO 3950 tooth numbering
- **Pathology Classification**: ResNet50 multi-label classifier (F1: 0.564)
- **Uncertainty Quantification**: Monte Carlo Dropout (T=30 passes)
- **Classes**: Caries, Deep Caries, Periapical Lesion, Impacted Tooth

## Usage
Upload a panoramic dental X-ray (OPG) in JPG or PNG format.

**Password**: Contact Dr. Enosh A. Paulson for access.

## Disclaimer
Research and educational use only. Not validated for clinical use.
Always consult a qualified dental professional for diagnosis.

## Author
Dr. Enosh A. Paulson  
BDS (RGUHS) | PGDMI Candidate, IIHMR Bangalore  
GitHub: [enosh729-design/Oral_guard](https://github.com/enosh729-design/Oral_guard)
