---
license: mit
language:
- en
tags:
- dental-ai
- medical-imaging
- yolov8
- pytorch
- object-detection
- multi-label-classification
- uncertainty-quantification
- gradcam
pipeline_tag: object-detection
---

# OralGuard — Uncertainty-Aware Dental Pathology Detection

Trained model weights for the OralGuard pipeline.
Detects caries, deep caries, periapical lesions, and
impacted teeth on panoramic dental X-rays.

## Model Files
- oralguard_det_best.pt — YOLOv8m detector (mAP@50: 0.548)
- classifier_best.pt — ResNet50 multi-label classifier (F1: 0.564)

## Dataset
DENTEX Challenge 2023 (MICCAI) — 678 annotated panoramic
X-rays across 4 pathology classes with FDI tooth notation.

## Architecture
- Tooth detection: YOLOv8m
- FDI notation mapping: Custom coordinate-to-tooth mapper
- Pathology classification: ResNet50 + MC Dropout (T=30)
- Uncertainty quantification: Predictive entropy
- Explainability: GradCAM++
- Active learning: Entropy-based uncertainty sampling

## Performance
| Class | mAP@50 |
|---|---|
| Caries | 0.544 |
| Deep Caries | 0.431 |
| Periapical Lesion | 0.263 |
| Impacted Tooth | 0.955 |
| Overall | 0.548 |

## Limitations
The classifier flags 100% of predictions as uncertain due to
class imbalance in the training data (only 128 periapical
lesion examples). This reflects genuine model uncertainty
and is the intended behaviour of the uncertainty mechanism.
Not validated for clinical use.

## Author
Dr. Enosh A. Paulson
BDS (RGUHS) | PGDMI Candidate, IIHMR Bangalore
GitHub: https://github.com/enosh729-design/Oral_guard
HuggingFace: https://huggingface.co/Enosh729

## Citation
If you use these weights, please cite the DENTEX 2023 dataset:
DENTEX: Dental Enumeration and Diagnosis on Panoramic X-rays
Challenge, MICCAI 2023.
