# `PerturbMerge`
Predicting the impact of **gene perturbations（repression)** on **cell survial and proliferation**.

PerturbMerge is a Python package that predicts the impact of ***485 specific gene knockouts*** on cell viability by analyzing single-cell transcriptomic data. It integrates **single-cell transcriptomic sequencing data** from 164 undisturbed cancer cell lines, along with **corresponding essential gene screening information**, and is trained using three ensemble learning algorithms: **Random Forest (RF)**, **Gradient Boosting Machine (GBM)**, and **XGBoost**. This results in a comprehensive classifier capable of predicting the effects of specific gene knockouts on the survival and proliferation of single cells based on the input data.

You can access the data used for PerturbMerge training and the pre-trained model that can be directly used for prediction through the following link: https://drive.google.com/drive/folders/1YwkYFRJbR4wEIdR1_8Dq1z1Y8_9FIXi0?usp=sharing
