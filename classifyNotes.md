{{ ... }}
# KrillBoost Classification Visualization Guide

This document provides a detailed explanation of the visualization features implemented in the KrillBoost package for evaluating and interpreting the krill presence/absence classification model.

## Overview of Visualization Features

The KrillBoost package includes several visualization tools designed to evaluate different aspects of the classification model's performance. These visualizations help in understanding model behavior, identifying potential issues, and communicating results effectively. All visualizations can be generated using the command-line interface:

```bash
python -m krillboost.plotting.plotPredict --figure [option]
```

Where `[option]` can be one of: `fig1`, `fig2`, `fig3`, `fig4`, `fig5`, `fig6`, `fig7`, or `all`.

## Detailed Explanation of Visualization Features

### 1. Confusion Matrix (fig1)

**What it shows:**
The confusion matrix provides a tabular summary of the model's predictions versus the actual observations. It displays four key metrics:
- True Positives (TP): Correctly predicted presence of krill
- True Negatives (TN): Correctly predicted absence of krill
- False Positives (FP): Incorrectly predicted presence when krill was absent
- False Negatives (FN): Incorrectly predicted absence when krill was present

**Scientific relevance:**
Confusion matrices are fundamental in species distribution modeling (SDM) literature as they provide a clear picture of model performance beyond simple accuracy metrics. In ecological studies, the costs of different types of errors may vary; for example, false negatives might be more problematic than false positives when identifying critical habitats for conservation. Studies by Elith & Leathwick (2009) and Franklin (2010) emphasize the importance of examining these error types separately in ecological modeling.

### 2. ROC Curve (fig2)

**What it shows:**
The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (sensitivity) against the False Positive Rate (1-specificity) at various threshold settings. The Area Under the Curve (AUC) provides a single metric summarizing the model's ability to discriminate between presence and absence, regardless of the classification threshold chosen.

**Scientific relevance:**
ROC curves are widely used in ecological modeling literature to evaluate model performance independently of the prevalence of the species in the training data. This is particularly important for krill studies where sampling efforts may be biased toward certain regions. The AUC value ranges from 0.5 (no discriminative ability, equivalent to random guessing) to 1.0 (perfect discrimination). In marine species distribution modeling, AUC values above 0.8 are typically considered good, while values above 0.9 are considered excellent (Fielding & Bell, 1997; Guisan et al., 2017).

### 3. Precision-Recall Curve (fig3)

**What it shows:**
The Precision-Recall curve plots precision (positive predictive value) against recall (sensitivity) at various threshold settings. The Average Precision (AP) summarizes the curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.

**Scientific relevance:**
Precision-Recall curves are particularly valuable for imbalanced datasets, which are common in ecological studies where presence data may be much less common than absence data. For krill distribution modeling, where absences might be more frequent than presences in certain regions, this visualization provides a more informative assessment of model performance than the ROC curve alone. Recent ecological modeling studies (e.g., Lobo et al., 2008; Mouton et al., 2010) have advocated for using Precision-Recall curves alongside ROC curves for a more comprehensive evaluation.

### 4. Probability Distribution (fig4)

**What it shows:**
This visualization displays the distribution of predicted probabilities for both actual presence and absence observations. It helps in understanding how well the model separates the two classes and in identifying appropriate probability thresholds for classification.

**Scientific relevance:**
Examining the distribution of predicted probabilities is crucial for understanding model calibration and for setting appropriate decision thresholds. In ecological literature, particularly in studies focusing on species range shifts under climate change (Elith et al., 2010), understanding the full distribution of probabilities rather than just binary predictions is essential for quantifying uncertainty. For krill distribution modeling, this visualization helps researchers understand the model's confidence in its predictions across different environmental conditions.

### 5. Calibration Curve (fig5)

**What it shows:**
The calibration curve (or reliability diagram) plots the observed frequency of the positive class against the predicted probability. A perfectly calibrated model would follow the diagonal line (y=x). Deviations from this line indicate that the predicted probabilities are either overconfident or underconfident.

**Scientific relevance:**
Calibration is particularly important in ecological modeling when the predicted probabilities are used for decision-making or risk assessment. In the marine ecology literature, well-calibrated probabilities are essential for quantifying uncertainty in predictions, especially when models are used to inform conservation or fisheries management decisions (Phillips & Elith, 2010). For krill studies, where predictions might inform CCAMLR (Commission for the Conservation of Antarctic Marine Living Resources) management decisions, proper calibration ensures that confidence levels in predictions accurately reflect actual occurrence rates.

### 6. Q-Q Plot (fig6)

**What it shows:**
The Quantile-Quantile (Q-Q) plot compares the distribution of predicted probabilities against a theoretical distribution (typically normal). This helps in assessing whether the model's predictions follow expected statistical patterns and in identifying potential outliers or systematic biases.

**Scientific relevance:**
Q-Q plots are less commonly reported in ecological modeling papers but are valuable for diagnostic purposes. They help identify whether model predictions follow expected statistical distributions, which is important for parametric statistical tests and for understanding model behavior. In advanced krill distribution studies, Q-Q plots can help identify regions or environmental conditions where the model might be systematically biased.

### 7. Spatial Predictions Heatmap (fig7)

**What it shows:**
This visualization creates a spatial heatmap of predicted krill presence probabilities across the study area. It combines:
- A color gradient representing the average predicted probability of krill presence in each grid cell
- Varying-sized black circles indicating the number of samples in each grid cell
- Bathymetry contours showing ocean depth at 400m intervals
- Land features and coastlines for geographical context

**Scientific relevance:**
Spatial prediction maps are perhaps the most widely used visualization in species distribution modeling literature, as they translate complex statistical models into intuitive geographical representations. The dual visualization approach (color for probability, circle size for sample density) addresses a common criticism in SDM literature regarding the lack of transparency about sampling effort and prediction confidence (Rocchini et al., 2011).

In Antarctic krill studies, similar spatial visualizations have been used by Atkinson et al. (2019) and Veytia et al. (2020) to show krill distribution patterns in relation to environmental variables. The addition of bathymetry contours is particularly relevant for krill, as their distribution is known to be influenced by ocean depth and proximity to the continental shelf (Murphy et al., 2007).

The heatmap approach, as opposed to point-based visualization, helps identify broader spatial patterns and regions of high krill probability while accounting for the inherent spatial autocorrelation in marine ecosystems. This visualization style is consistent with CCAMLR reporting standards and facilitates comparison with other krill distribution studies.

## Interpretation in the Context of Two-Step Modeling

It's important to note that these visualizations specifically evaluate the first step of our two-step modeling approach: the presence/absence classification. As documented in our data transformation process, the original `STANDARDISED_KRILL_UNDER_1M2` values have been transformed into:

1. `KRILL_PRESENCE`: Binary indicator (1 or 0) of krill presence/absence
2. `KRILL_LOG10`: Log10-transformed abundance values where krill is present

The visualizations in this document focus on evaluating how well our model predicts the `KRILL_PRESENCE` variable. A separate set of visualizations would be needed to evaluate the second step of the model, which predicts `KRILL_LOG10` (abundance) only where krill is present.

## References

Atkinson, A., Hill, S. L., Pakhomov, E. A., Siegel, V., Reiss, C. S., Loeb, V. J., ... & Sailley, S. F. (2019). Krill (Euphausia superba) distribution contracts southward during rapid regional warming. Nature Climate Change, 9(2), 142-147.

Elith, J., & Leathwick, J. R. (2009). Species distribution models: ecological explanation and prediction across space and time. Annual Review of Ecology, Evolution, and Systematics, 40, 677-697.

Elith, J., Kearney, M., & Phillips, S. (2010). The art of modelling range‐shifting species. Methods in Ecology and Evolution, 1(4), 330-342.

Fielding, A. H., & Bell, J. F. (1997). A review of methods for the assessment of prediction errors in conservation presence/absence models. Environmental Conservation, 24(1), 38-49.

Franklin, J. (2010). Mapping species distributions: spatial inference and prediction. Cambridge University Press.

Guisan, A., Thuiller, W., & Zimmermann, N. E. (2017). Habitat suitability and distribution models: with applications in R. Cambridge University Press.

Lobo, J. M., Jiménez‐Valverde, A., & Real, R. (2008). AUC: a misleading measure of the performance of predictive distribution models. Global Ecology and Biogeography, 17(2), 145-151.

Mouton, A. M., De Baets, B., & Goethals, P. L. (2010). Ecological relevance of performance criteria for species distribution models. Ecological Modelling, 221(16), 1995-2002.

Murphy, E. J., Watkins, J. L., Trathan, P. N., Reid, K., Meredith, M. P., Thorpe, S. E., ... & Fleming, A. H. (2007). Spatial and temporal operation of the Scotia Sea ecosystem: a review of large-scale links in a krill centred food web. Philosophical Transactions of the Royal Society B: Biological Sciences, 362(1477), 113-148.

Phillips, S. J., & Elith, J. (2010). POC plots: calibrating species distribution models with presence‐only data. Ecology, 91(8), 2476-2484.

Rocchini, D., Hortal, J., Lengyel, S., Lobo, J. M., Jiménez-Valverde, A., Ricotta, C., ... & Chiarucci, A. (2011). Accounting for uncertainty when mapping species distributions: the need for maps of ignorance. Progress in Physical Geography, 35(2), 211-226.

Veytia, D., Corney, S., Meiners, K. M., Kawaguchi, S., Murphy, E. J., & Bestley, S. (2020). Circumpolar projections of Antarctic krill growth potential. Nature Climate Change, 10(6), 568-575.
{{ ... }}