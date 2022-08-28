#### Given:

- CT scans with approach 30 slices each
- CSV file with slice-level annotations

'ANY' column: 1 means slice contains haemorrgage, 0 means doesn't

IPH,IVH,SAH,SDH: four subtypes of haemorrhages. Could have multiple at the same time

---

#### Evaluation:

Sensitivity and specificity (exists or not in the brain), 1 or 0. Subtypes do not
matter for now. If they did, multiclass classification? One scan could have multiple classes tho

---

#### Thoughts:

- Image Folder format didn't quite work, PyTorch doesn't support the extension
(should have checked first haha)

- Wrote custom data loader + all related code

- Started experimenting. Things to play with off the top of my head:

    - SOTA model / custom
    - Pretrained SOTA / not pretrained 
    - Fine tuning SOTA or just training the head
    - Augmentation (which one?) / no augmentation
    - Balancing classes (imbalanced dataset) / can't be bothered
    - Batch size (14 images loads my gpu only ~ 40% - 24 GB VRAM)
    - LR
    - Optimizer
    - Loss function (i doubt there're many options though)
    - Epochs
    - 
    
---

#### Some results:

```
calculate_metrics 85: TP: 156; TN: 1566; FP: 104; FN: 115
calculate_metrics 87: Sensitivity: 0.5756; Specificity: 0.9377
```