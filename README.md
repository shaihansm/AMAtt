
# AMAtt: Manifold Attention Network with Adaptive Log-Euclidean Metrics

### EEG Signal Decoding using Geometric Deep Learning

**Authors**:  
S. M. Syed A. Shaihan, Imad Tibermacine, Christian Napoli  
Sapienza Università di Roma, Italy  

---

## 🚀 Overview

**AMAtt** is a novel deep learning model for decoding **EEG signals** using **Riemannian geometry** and **manifold attention**. It extends the **MAtt (Manifold Attention Network)** by introducing **Adaptive Log-Euclidean Metrics (ALEM)**—a learnable metric that improves EEG representation on the Symmetric Positive Definite (SPD) manifold.

---

##  Key Features

-  **Adaptive Log-Euclidean Metric (ALEM)** for learning optimal geometry
-  Robust spatiotemporal feature extraction from EEG data
-  Improved EEG classification performance over baseline MAtt
-  Evaluated on two benchmark datasets: **BCIC-IV-2a** (motor imagery) and **MAMEM-SSVEP-II** (SSVEP)

---

## Results Summary

| Dataset        | Model  | Test Accuracy |
|----------------|--------|----------------|
| BCIC-IV-2a     | MAtt   | 57.64%         |
|                | AMAtt  | **63.19%**     |
| MAMEM-SSVEP-II | MAtt   | 41.40%         |
|                | AMAtt  | **46.00%**     |

---

## Datasets Used

- **BCIC-IV-2a**: 4-class motor imagery EEG dataset  
- **MAMEM-SSVEP-II**: Time-synchronous SSVEP EEG signals  

---

## Architecture

1. **Feature Extraction**: CNN-based encoder  
2. **E2R Layer**: Projects features onto SPD manifold  
3. **Manifold Attention Module**: Captures temporal dependencies on the manifold  
4. **ALEM**: Learns data-adaptive metrics for better SPD representation  
5. **R2E Layer**: Maps features back for final classification  

---

---

## License

This work is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

---

## 🧾 Citation

If you use this work, please cite:

```text
S. M. Syed A. Shaihan, Imad Tibermacine, Christian Napoli. 
"AMAtt: Manifold Attention Network with Adaptive Log-Euclidean Metrics", 2025.
