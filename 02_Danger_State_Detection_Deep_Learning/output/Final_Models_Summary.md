# Cobot Safety Models Evaluation
**Evaluation Parameters:**
- **Strategy:** Factory-Calibrated (Subject-Dependent)
- **Post-Processing:** Median Filter (Temporal Smoothing)
- **Fixed Decision Threshold:** 0.77

Models are ranked below from best to worst based on their **Macro F1 Score**.

---

## 1. ConvNeXt 1D
- **Accuracy:** 99.43%
- **Macro F1 Score:** 98.68%
- **Danger Recall:** 97.13%
- **Danger Precision:** 98.25%

### Confusion Matrix
![ConvNeXt 1D Confusion Matrix](./ConvNeXt_1D_cm.png)

| | Predicted SAFE | Predicted DANGER |
|---|---|---|
| **Actual SAFE** | 60,411 (True Negatives) | 148 (False Positives) |
| **Actual DANGER** | 245 (False Negatives) | 8,290 (True Positives) |

---

## 2. TCN
- **Accuracy:** 99.41%
- **Macro F1 Score:** 98.62%
- **Danger Recall:** 97.14%
- **Danger Precision:** 98.04%

### Confusion Matrix
![TCN Confusion Matrix](./TCN_cm.png)

| | Predicted SAFE | Predicted DANGER |
|---|---|---|
| **Actual SAFE** | 60,393 (True Negatives) | 166 (False Positives) |
| **Actual DANGER** | 244 (False Negatives) | 8,291 (True Positives) |

---

## 3. MLP-Mixer 1D
- **Accuracy:** 99.28%
- **Macro F1 Score:** 98.32%
- **Danger Recall:** 95.88%
- **Danger Precision:** 98.27%

### Confusion Matrix
![MLP-Mixer 1D Confusion Matrix](./MLP-Mixer_1D_cm.png)

| | Predicted SAFE | Predicted DANGER |
|---|---|---|
| **Actual SAFE** | 60,415 (True Negatives) | 144 (False Positives) |
| **Actual DANGER** | 352 (False Negatives) | 8,183 (True Positives) |

---

## 4. InceptionTime
- **Accuracy:** 98.30%
- **Macro F1 Score:** 96.22%
- **Danger Recall:** 97.53%
- **Danger Precision:** 89.62%

### Confusion Matrix
![InceptionTime Confusion Matrix](./InceptionTime_cm.png)

| | Predicted SAFE | Predicted DANGER |
|---|---|---|
| **Actual SAFE** | 59,595 (True Negatives) | 964 (False Positives) |
| **Actual DANGER** | 211 (False Negatives) | 8,324 (True Positives) |

---

## 5. Transformer 1D
- **Accuracy:** 96.47%
- **Macro F1 Score:** 92.50%
- **Danger Recall:** 95.85%
- **Danger Precision:** 79.71%

### Confusion Matrix
![Transformer 1D Confusion Matrix](./Transformer_1D_cm.png)

| | Predicted SAFE | Predicted DANGER |
|---|---|---|
| **Actual SAFE** | 58,477 (True Negatives) | 2,082 (False Positives) |
| **Actual DANGER** | 354 (False Negatives) | 8,181 (True Positives) |

---

