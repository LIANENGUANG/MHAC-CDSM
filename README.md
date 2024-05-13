# MHAC-CDSM: Multi-Head Attention-based Cross-modal Diagnosis and Staging Model with Pseudo-Image Approach

This repository contains the implementation of the Multi-Head Attention-based Cross-modal Diagnosis and Staging Model (MHAC-CDSM) for predicting tumor stage using multi-omics data, including clinical data, miRNA expression, mRNA expression, and DNA methylation data. The model incorporates multi-head self-attention mechanisms to capture the interactions and dependencies among different data modalities and utilizes a pseudo-image approach with 1x1 convolution to process DNA methylation data.

## Dataset

The dataset used in this project can be accessed via the following Google Drive link:[Dataset Link](https://drive.google.com/drive/folders/100kREKOJBSrByy_A693x3xdzLqa9HcUL?usp=drive_link)

Please download the dataset and place it in the appropriate directories as described in the "Directory Structure" section.

## Directory Structure

The project has the following directory structure:

```shell
.
├── README.md  
├── data
│   ├── resampled_clinical_mirna_mina_data
│   │   ├── resampled_clinical_data.csv
│   │   ├── resampled_mirna_data.csv  
│   │   └── resampled_mrna_data.csv
│   └── resampled_methy_data_by_chrom
│       ├── resampled_methy_data_chr1.csv
│       ├── resampled_methy_data_chr2.csv
│       ├── ...  
│       └── resampled_methy_data_chrX.csv
├── model
│   ├── FCNN-CDSM.ipynb
│   └── MHAC-CDSM.ipynb
└── requirements.txt
```

- The `data` directory contains the resampled clinical, miRNA, mRNA, and DNA methylation data.
- The `model` directory contains the Jupyter Notebook files for the MHAC-CDSM and FCNN-CDSM models.  
- The `requirements.txt` file specifies the required dependencies for the project.

## Requirements

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage

1. Download the dataset from the provided Google Drive link and place the files in the appropriate directories as described in the "Directory Structure" section.

2. Install the required dependencies by running `pip install -r requirements.txt`.

3. Open the Jupyter Notebook files (`MHAC-CDSM.ipynb` and `FCNN-CDSM.ipynb`) in the `model` directory to explore and run the models.

4. Follow the instructions in the notebook files to preprocess the data, train the models, and evaluate their performance.

## Model Architecture

<img src="../../Typora/model_architecture.tiff" alt="model_architecture" style="zoom: 67%;" />

The MHAC-CDSM model consists of the following components:

- **Multi-Head Attention Layers**: These layers are used to adjust the feature dimensions of the DNA methylation, miRNA, and mRNA data. The number of attention heads and target dimension can be configured.

- **Pseudo-Image Approach**: The DNA methylation data across different chromosomes is treated as a pseudo-image, and a 1x1 convolutional layer is applied to extract compressed features.

- **Data Integration**: The adjusted features from DNA methylation, miRNA, and mRNA data are concatenated along with the clinical data to form the final feature set.

- **Training and Evaluation**: The integrated feature set is split into training and test sets. The model is trained on the training set and evaluated on the test set using appropriate metrics.

The FCNN-CDSM model serves as a baseline model for comparison.

## Results

This section presents the performance comparison between our proposed MHAC-CDSM model and the baseline model, FCNN-CDSM. We evaluate the models using various machine learning algorithms, including K-Nearest Neighbors (KNN), Random Forest (RF), XGBoost, CatBoost, Neural Network (NN), and Convolutional Neural Network (CNN). The performance is measured using several metrics, such as Accuracy, Macro Precision, Micro Precision, Macro Recall, Micro Recall, Macro F1-score, and Micro F1-score.

![model_comparison](../../Typora/model_comparison-5610096.tiff)

### FCNN-CDSM (Baseline Model)

| Model    | Accuracy | Macro Precision | Micro Precision | Macro Recall | Micro Recall | Macro F1-score | Micro F1-score |
| -------- | -------- | --------------- | --------------- | ------------ | ------------ | -------------- | -------------- |
| KNN      | 0.8198   | 0.8383          | 0.8198          | 0.8227       | 0.8198       | 0.8273         | 0.8198         |
| RF       | 0.9932   | 0.9945          | 0.9932          | 0.9920       | 0.9932       | 0.9932         | 0.9932         |
| XGBoost  | 0.9617   | 0.9699          | 0.9617          | 0.9554       | 0.9617       | 0.9596         | 0.9617         |
| CatBoost | 0.9640   | 0.9726          | 0.9640          | 0.9573       | 0.9640       | 0.9618         | 0.9640         |
| NN       | 0.7191   | 0.7233          | 0.7191          | 0.7152       | 0.7191       | 0.7117         | 0.7191         |
| CNN      | 0.6404   | 0.6694          | 0.6404          | 0.6515       | 0.6404       | 0.6484         | 0.6404         |

### MHAC-CDSM (Our Model)

| Model    | Accuracy | Macro Precision | Micro Precision | Macro Recall | Micro Recall | Macro F1-score | Micro F1-score |
| -------- | -------- | --------------- | --------------- | ------------ | ------------ | -------------- | -------------- |
| KNN      | 0.9459   | 0.9452          | 0.9459          | 0.9449       | 0.9459       | 0.9450         | 0.9459         |
| RF       | 0.9955   | 0.9964          | 0.9955          | 0.9947       | 0.9955       | 0.9955         | 0.9955         |
| XGBoost  | 0.9730   | 0.9790          | 0.9730          | 0.9680       | 0.9730       | 0.9718         | 0.9730         |
| CatBoost | 0.9685   | 0.9757          | 0.9685          | 0.9627       | 0.9685       | 0.9668         | 0.9685         |
| NN       | 0.8539   | 0.8544          | 0.8539          | 0.8505       | 0.8539       | 0.8506         | 0.8539         |
| CNN      | 0.8315   | 0.8270          | 0.8315          | 0.8235       | 0.8315       | 0.8036         | 0.8315         |

The results demonstrate that our MHAC-CDSM model outperforms the FCNN-CDSM baseline across all evaluation metrics and machine learning algorithms. The Random Forest (RF) model in our MHAC-CDSM approach achieves the highest performance, closely followed by XGBoost and CatBoost.

In summary, our MHAC-CDSM model demonstrates exceptional performance, particularly with the Random Forest, XGBoost, and CatBoost models, outperforming the FCNN-CDSM baseline in all cases. The multi-head attention mechanism and pseudo-image approach contribute to the model's ability to capture relevant features and patterns for accurate cancer stage prediction.

These results highlight the effectiveness of our proposed MHAC-CDSM model in improving cancer stage prediction performance compared to the baseline approach. The incorporation of attention mechanisms and the pseudo-image approach enables our model to learn more discriminative and robust representations, leading to enhanced prediction accuracy.

## Future Work

To further enhance our MHAC-CDSM model, we plan to focus on the following key areas:

### 1. Model Architecture Optimization

- Explore advanced attention mechanisms and alternative architectures
- Investigate the integration of graph neural networks (GNNs) and transformer-based models

### 2. Multimodal Data Fusion

- Incorporate imaging data (CT, MRI, PET scans) and pathology data (histology, immunohistochemistry)
- Develop advanced multimodal fusion techniques, such as feature-level and decision-level fusion

### 3. Interpretability and Visualization

- Improve model interpretability to enhance trust and usability in clinical decision support
- Utilize attention visualization, feature importance analysis, and model explanation techniques

Through these research directions, we aim to develop a powerful and robust MHAC-CDSM model that can assist healthcare professionals in accurate cancer stage prediction and improve patient outcomes.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or inquiries, please contact [lianenguang@gmail.com](mailto:lianenguang@gmail.com).

In conclusion, the MHAC-CDSM model presented in this repository demonstrates the effectiveness of multi-head attention mechanisms and the pseudo-image approach for improving cancer stage prediction using multi-omics data. The model's ability to capture complex interactions and dependencies among different data modalities leads to enhanced prediction performance compared to the baseline FCNN-CDSM model.

We hope that this work will contribute to the advancement of cancer diagnosis and staging, ultimately benefiting patients and healthcare providers. We welcome collaborations, feedback, and suggestions from the research community to further improve and extend our model.

Thank you for your interest in our project!
