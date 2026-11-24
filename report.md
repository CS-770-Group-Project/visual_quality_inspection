---
title: "Visual Quality Inspection of Cast Parts"
subtitle: "CS770 Machine Learning, Team Project"
author: "Supriya Gurung, Jennifer Isobe, Sydney Epke"
geometry: margin=0.6in
fontsize: 10pt
header-includes: |
  \usepackage{float}
  \floatplacement{figure}{H}
  \usepackage[font=small,skip=3pt]{caption}
  \AtBeginEnvironment{longtable}{\small}
  \AtBeginDocument{\setlength{\parskip}{2pt plus 1pt}}
  \setlength{\LTpre}{2pt}
  \setlength{\LTpost}{2pt}
  \setlength{\textfloatsep}{6pt}
  \setlength{\intextsep}{4pt}
  \AtBeginDocument{\Urlmuskip=0mu plus 1mu}
  \makeatletter
  \renewcommand{\@maketitle}{\begin{center}{\Large\bfseries\@title\par}\vspace{3pt}{\normalsize\@author\par}\end{center}\vspace{-6pt}}
  \renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}{-1.6ex plus -0.4ex}{0.5ex}{\normalfont\large\bfseries}}
  \makeatother
---

## Problem and data

This project involves classifying parts as defective (1) or acceptable (0) using the original casting_512x512 collection [1]. This collection contains 1,300 top-down 512x512 RGB images of cast parts of which 781 are defective and 519 are acceptable. An audit checking for unreadable files and duplicate images, checked via hashing decoded pixels, found nothing so no images were excluded. The original collection was split using a stratified 70/15/15 split with seed 770 resulting in 910 training, 195 validation, and 195 test images. The validation and testing set each had 117 defective and 78 acceptable images. Each model tested uses the same input of grayscale 64x64 images scaled to [0,1], with no augmentation or learned normalization.

## Models

**Baseline:** scikit-learn logistic regression [2] on the 4,096 flattened pixels (lbfgs, C = 1.0, no class weighting, tolerance 1e-6). The default tolerance (1e-4) stopped early at a machine-dependent point, so team members' results differed by one image; with 1e-6 it converges fully (643 iterations, no warnings).

**CNN:** A small PyTorch [3] network with two convolutional blocks (3x3 convolution -> ReLU -> 2x2 max-pooling, with 16 and 32 filters), followed by a 256-unit fully connected layer with ReLU and dropout, and one output (a logit for "defective"). We trained with binary cross-entropy on logits, Adam [4] (learning rate 0.001), batch size 32, and 15 epochs, and used the weights after the final epoch. The network has 2.1 million parameters, almost all in the first fully connected layer (8,192 -> 256). Training ran on the CPU (Python 3.12.14, PyTorch 2.15.0, macOS); seeds were reset to 770 before each run.

**Controlled experiment:** CNN B is identical to CNN A except for the dropout probability, which is 0.25 instead of 0.0 [5]. We expected dropout to reduce overfitting and possibly improve validation F1, at some risk to recall. That did not happen: CNN B scored lower on validation F1 (0.864 vs. 0.876) and recall (0.897 vs. 0.906). Its final training loss was higher (0.205 vs. 0.162), as expected with dropout, but its validation loss was not lower (0.366 vs. 0.360). For both runs, validation loss levels off around 0.36 after about epoch 12 while training loss keeps falling (Figure 1). With one seed (770) and 195 validation images, the 0.012 difference in F1 amounts to about two predictions.

## Validation comparison and selection

We selected the CNN A model according to validation F1 for the defective class, with a fixed rule that P(defective) $\geq$ 0.5 means defective and ties going to the simpler model. The test set was not used until the selection was final and the notebook re-checks the frozen model before testing it.

| Configuration | Precision | Recall | F1 | Accuracy | Missed defects | Good parts rejected | Training time (s) |
|:--------------------------|------:|------:|-----:|-------:|--------:|---------:|---------:|
| Logistic regression | 0.861 | 0.846 | 0.853 | 0.826 | 18 | 16 | 1.0 |
| CNN A (no dropout) | 0.848 | 0.906 | 0.876 | 0.846 | 11 | 19 | 14.4 |
| CNN B (dropout 0.25) | 0.833 | 0.897 | 0.864 | 0.831 | 12 | 21 | 12.5 |

: Validation results (n = 195). Precision, recall, and F1 are for the defective class.

![Training and validation loss by epoch for both CNN runs.](report_figures/loss_curves.png){width=80%}

## Test results for the selected model

```{=latex}
\noindent
\begin{minipage}[c]{0.64\linewidth}
\centering
\captionof{table}{Test results (n = 195), defective class.}
{\small
\begin{tabular}{lrrrr}
\toprule
Model & Precision & Recall & F1 & Accuracy \\
\midrule
CNN A (no dropout) & 0.864 & 0.974 & 0.916 & 0.892 \\
\bottomrule
\end{tabular}}

\vspace{4pt}
\raggedright
The model correctly identified 114 of 117 defective castings (3 false negatives, 2.6\%) and rejected 18 of 78 acceptable castings (false positives, 23\%). Even though test F1 (0.916) is higher than validation F1 (0.876), with 195 images per set this reflects sampling variation, not an improvement, and nothing was changed after testing.
\end{minipage}\hfill
\begin{minipage}[c]{0.32\linewidth}
\centering
\includegraphics[width=\linewidth]{report_figures/test_confusion_matrix.png}
\captionof{figure}{Test confusion matrix.}
\end{minipage}
```

**Error analysis:** For this model application, false negatives are more critical than false positives. In quality inspection for machine parts, missing a defect (false negative) can result in the part failing in service or lead to a recall. False positives run the risk of additional cost associated with rechecking the part or scrapping a good part, but this is likely to be less costly than part failures and recalls in the long run. The chosen model (CNN A) fits this priority with a recall of 0.974, but flagged part(s) will still need a manual recheck. In examining the missed defects, 2 of the 3 were close to the established threshold with P=0.38 and P=0.45. The visible defects in their images are small marks on the rim or notches that are only one or two pixels at the 64x64 images size. The other missed defect has P=0.016 and has no obvious defects on the part but has a flat gray background. This suggests that the model may be relying on the background and/or lighting for its classification. Images with the most confident false positives (P=0.77-0.90) have bright reflections and visible machining lines that, at low resolution, may appear to be defects.

## Limitations (summarized from project.ipynb)

1. **Controlled photographs from one source:** All images show one casting part design with similar lighting, background, and framing from a top down view. This limited variation limits the applicability of the trained model so, these results do not support claims about a new factory, product, camera, or setup without new labeled data associated with each.
2. **Small evaluation sets and a single run:** Due to the small validation and testing dataset size, one prediction changes F1 by about 0.005. Each model was trained once with one seed and CNN results varied slightly between computers so some differences between the models may have been masked.
3. **Unrealistic class balance:** Defects make up 60% of this dataset but in a real production line, this percentage would be much lower. This would correlate to a lower precision so more acceptable parts would be rejected.
4. **Low resolution and exact-only duplicate checks:** Downsizing to 64x64 grayscale images can hide small defects and near but not exact duplicate photos of one part could appear in more than one split.

## References

[1] R. Dabhi, *Casting Product Image Data for Quality Inspection*, Kaggle.
<https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product>

[2] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR* 12, 2011.

[3] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *NeurIPS*, 2019.

[4] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," *ICLR*, 2015.

[5] N. Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," *JMLR* 15, 2014.

## Team contributions

**Supriya Gurung:** Project setup, data audit, split, preprocessing, logistic regression baseline, report.

**Sydney Epke:** CNN design and training, dropout experiment, validation comparison, loss curves, model selection, report.

**Jennifer Isobe:** Selection checks, test evaluation and error analysis, review against the guide, reproducibility fixes, notebook merging, discussion, report.

## AI-assistance disclosure

We used Claude (Anthropic) via Claude Code to review the repository against the project guide; diagnose and suggest reproducibility issue fixes; draft the selection-check and test-evaluation code; merge the team's notebooks; and draft layouts for the notebook explanations, discussion, limitations, conclusions, and an outline for the report, which the team then reviewed and edited. The team made the experiment decisions and reviewed, verified, and can explain all submitted code and text.
