# RSNA Screening Mammography Breast Cancer Detection
#### Find breast cancers in screening mammograms
This is a Kaggle competition that challenges data scientists to use images to identify breast cancer. The below description is taken from the [competition host website](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview).

## Goal of the Competition

The goal of this competition is to identify breast cancer. You'll train your model with screening mammograms obtained from regular screening.

Your work improving the automation of detection in screening mammography may enable radiologists to be more accurate and efficient, improving the quality and safety of patient care. It could also help reduce costs and unnecessary medical procedures.

## Context

According to the WHO, breast cancer is the most commonly occurring cancer worldwide. In 2020 alone, there were 2.3 million new breast cancer diagnoses and 685,000 deaths. Yet breast cancer mortality in high-income countries has dropped by 40% since the 1980s when health authorities implemented regular mammography screening in age groups considered at risk. Early detection and treatment are critical to reducing cancer fatalities, and your machine learning skills could help streamline the process radiologists use to evaluate screening mammograms.

Currently, early detection of breast cancer requires the expertise of highly-trained human observers, making screening mammography programs expensive to conduct. A looming shortage of radiologists in several countries will likely worsen this problem. Mammography screening also leads to a high incidence of false positive results. This can result in unnecessary anxiety, inconvenient follow-up care, extra imaging tests, and sometimes a need for tissue sampling (often a needle biopsy).

The competition host, the Radiological Society of North America (RSNA) is a non-profit organization that represents 31 radiologic subspecialties from 145 countries around the world. RSNA promotes excellence in patient care and health care delivery through education, research, and technological innovation.

Your efforts in this competition could help extend the benefits of early detection to a broader population. Greater access could further reduce breast cancer mortality worldwide.

## Evaluation
Submissions are evaluated using the probabilistic F1 score (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. You can find a Python implementation here.

With pX as the probabilistic version of X:

```math
pF_1 = 2\frac{pPrecision \cdot pRecall}{pPrecision+pRecall}
```

where:

```math
pPrecision = \frac{pTP}{pTP+pFP}
```
```math
pRecall = \frac{pTP}{TP+FN}
```

## Submission Format
For each prediction_id, you should predict the likelihood of cancer in the corresponding cancer column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
1-L,1
...
```
