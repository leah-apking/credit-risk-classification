## Supervised Machine Learning

### Credit Risk Analysis Overview

In this analysis I evaluated the creditworthiness of borrows based on a dataset containing historical lending activity from a peer-to-peer lending services company. This data was used to train a supervised machine learning logistic regression model to predict whether loans to the borrowers are high-risk or healthy loans.

The dataset used for this model included loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks, total debt, and loan status. The loan status value indicated the risk of each borrower. With the goal of predicting credit risk this became our target value, the remaining descriptive information was used to predict this target. The vast majority of borrowers in this dataset were identified as low/no risk, with high-risk borrowers making up only `3.3` percent.

After splitting the data in to training and testing sets, I created a logistic regression model, fit it to the training data, and then used it to predict the risk of the test data. This produced a balanced accuracy score of `0.952`, however the confusion matrix and classification report showed that while the model was able to correctly predict the low/no risk loans, it struggled to catch all of the high-risk borrowers. Given that the goal was to predict high-risk loans correctly, the model was underperforming.

To address this, I used RandomOverSampler from imbalanced-learn to increase the number of high-risk borrowers for the model to train on. Once the data was resampled it had an equal number of high-risk and low/no risk loans to train on. I repeated the progress for this logistic regression model, but used the new training data for fitting. The evaluation of this model’s performance in predicting the original test data showed significant improvement.

### Results

* Machine Learning Model with Original Data:
  * **Balanced Accuracy Score**: `0.952`
  * **Precision**: Low/no risk = `1.00`; high-risk = `0.85`
  * **Recall**: Low/no risk = `0.99`; high-risk = `0.91`

* Machine Learning Model with RandomOverSampling:
  * **Balanced Accuracy Score**: `0.994`
  * **Precision**: Low/no risk = `1.00`; high-risk = `0.84`
  * **Recall**: Low/no risk = `0.99`; high-risk = `0.99`


### Summary

Both supervised machine learning models performed well and had accuracy scores over 0.95. However, given that this model will likely be used by lenders to determine the creditworthiness of individual borrowers, the model’s usefulness is in its ability to correctly predict high-risk borrowers. This means we need to pay close attention to the model’s high-risk recall score.

The logistic regression model using the RandomOverSampler with equal value counts was more accurate than the model using only the original data. It continued correctly classify healthy loans extremely well and improved on classifying high risk loans. The increase in high-risk testing data allowed this model to correctly classify nearly all of high-risk loans, missing only 4 of the 619 high-risk loans in the dataset. However, while it did well in true positives and true negatives, with high recall for high-risk loans it most likely to misclassify a healthy loan as a high-risk loan. Given the application of this would be to identify high-risk loans, this model would correctly identify nearly all of them, with the most likely errors being a misclassification of a healthy borrower.

