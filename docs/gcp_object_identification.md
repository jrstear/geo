# GCP Object Identification Strategy (Orange Clay Pigeons)

This document outlines the recommended object identification approach for Ground Control Points (GCPs) structured as an "X" of orange clay pigeons in drone imagery.

## The Hybrid Pipeline Strategy

Given the tiny size of GCPs in high-resolution drone imagery and the distinct color signature of orange clay pigeons, a three-stage hybrid pipeline is recommended.

### Stage 1: Color-based Candidate Detection (Pre-filter)
*   **Method**: Convert imagery to HSV color space and use `cv2.inRange()` to isolate the specific orange pigment.
*   **Goal**: Eliminate 99% of the pixels (grass, dirt, water) in milliseconds to identify high-probability "clusters".

### Stage 2: Geometric Refinement (Shape Check)
*   **Method**: Use `cv2.findContours` on the orange masks. Look for clusters of 4-5 blobs arranged in a cross/X geometry.
*   **Goal**: Distinguish intentional patterns from incidental orange noise (e.g., orange leaves or trash).

### Stage 3: Lightweight CNN/DNN Verification (Final ID)
*   **Method**: Train a custom Convolutional Neural Network (CNN) or a small YOLO (v8-nano) model specifically on cropped sub-images (e.g., 128x128 pixels).
*   **Goal**: Confidently classify the cluster as a GCP target.

## Comparison of OpenCV Approaches

| Approach | Accuracy | Compute | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- | :--- |
| **Haar Cascades** | Medium-Low | Very Low | Fast, standard. | Grayscale only; high false-positive rate. |
| **Template Matching**| High | Low | Simple. | Fails on rotation and scale changes. |
| **Custom CNN/DNN** | **Very High** | High | Robust; learns color + shape. | Requires labeled training data. |

## Training Data Requirements (geo-bwb)

*   **Positive Samples**: Tightly cropped (5-10% padding) square images of the "X" targets.
*   **Negative Samples**: "Hard negatives" containing similar-colored non-target objects or typical terrain features.
*   **Resolution**: Must preserve Ground Sample Distance (GSD) to ensure pigeons are discernible.
