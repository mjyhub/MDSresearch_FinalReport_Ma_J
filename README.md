# Sustained Mask Mandates and High Mask-Wearing Behaviour

## Project overview

This repository contains the materials used for the Part B United States extension of the mask-wearing research project.

The study examines the relationship between sustained mask mandate periods and high mask-wearing behaviour across the United States and Australia during the COVID-19 pandemic.

The project addresses the following research questions:

1. How do high mask-wearing rates vary across US states before and during sustained mandate periods?
2. Is a sustained mask mandate period associated with high mask-wearing behaviour in the United States?
3. Does the association between sustained mandate periods and high mask-wearing behaviour differ between Australia and the United States?

## Repository structure

The main project materials are stored in the `PartB_US_Extension` folder:

- `code/`: R and Python scripts used for data preparation, statistical analysis and visualisation.
- `data/`: cleaned and processed datasets used in the analyses.
- `raw_data/`: original data files used to construct the analysis datasets.
- `results/`: generated figures, tables and statistical model outputs.

- 
## Figures used in the final report

The following figures were generated for and included in the final report:

| File | Description |
|---|---|
| `design_overview_4.pdf` | Presents the analytical framework and overall workflow of the study. |
| `02_state_level_descriptive_comparison.png` | Compares changes in high mask-wearing rates across US states before and during sustained mask mandate periods. |
| `04c_rq2_holdout_or_forest_plot.png` | Presents the odds ratio estimates and confidence intervals from the RQ2 hold-out robustness analysis. |
| `06c_rq3_holdout_interaction_forest_plot.png` | Presents the interaction estimates and confidence intervals from the RQ3 Australia–US hold-out robustness analysis. |

These figures were generated using the analysis scripts provided in the `code` folder.

## Analysis workflow

The main analysis workflow includes:

1. processing the US mask mandate policy data;
2. identifying sustained mask mandate periods using a 14-day rolling average;
3. cleaning and preparing the US survey data;
4. constructing the high mask-wearing outcome variable;
5. conducting state-level descriptive comparisons;
6. fitting the US logistic regression model;
7. fitting the Australia–US interaction model; and
8. conducting hold-out robustness analyses.

The analyses use:

- YouGov COVID-19 behavioural survey data; and
- Oxford COVID-19 Government Response Tracker policy data.

## Author

Jiayi Ma  
Master of Data Science  
The University of Adelaide
