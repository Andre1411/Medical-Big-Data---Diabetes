# Diabetes onset, 10 years follow-up 
This is the code for a University project for the course of Medical Big Data Sources and Clinical Decision Support Systems (accademic year 2021/2022).
## Description
### Objective
The purpose of the project work is to develop and validate a predictive model that forecasts the diabetes status at 10 years
(outcome) using an optimal subset of the variables available at baseline.
### Dataset:
The dataset is provided in the dataframe *"data"* stored in the file *"data_final_project.RData"*.
The dataframe includes 24 variables (columns of the dataframe) collected for 3,264 subjects
(rows of the dataframe).
All the variables, except the outcome, were collected at the baseline time, i.e. the enrolment
of the subject in the study. At the baseline time all the subjects did not have diabetes. During
the follow-up period of 10 years, some of the subjects developed diabetes. The variable
‘outcome’ represents the diabetes status of each patient 10 years after the baseline:
- Outcome = 0 → 10 years after the baseline the subject did not have diabetes
- Outcome = 1 → 10 years after the baseline the subject had diabetes

All the available variables are described in the following table:

| Variable | Description | Values |
| - | - | - |
| gender | Gender/sex | 0=females, 1=males |
| age | Age | Integers [years] |
| education | Education level | 0=grade 11 or less, 1=completed high school, 2=bachelor degree, 3=master degree |
| marital_status | Marital status | 0 = married/living as married, 1 = divorced/separated/widowed, 2= never married |
| smoking | Smoking status | 0 = no, 1 = past smoker, 2 = current smoker |
| bmi | Body mass index | Double [kg/m^2] |
| waist | Waist circumference | Double [cm] |
| depression_scale | CESD Depression scale| Integers, range 0-8, 0 = no depression symptoms, 8 = max level of depression symptoms |
| mod_vig_pa | Frequency of moderate and vigorous physical activity | 0 = hardly ever or never, 1 = 1-3 times per month, 2 = once per week, 3 = more than once per week |
| heartd_hx | Ever had doctor diagnosed heart disease | 0 = no, 1 = yes |
| hypertension | Ever had doctor diagnosed hypertension | 0 = no, 1 = yes |
| high_chol | Ever had doctor diagnosed high cholesterol | 0 = no, 1 = yes |
| sr_poor_health | Self-reported health status | 0 = excellent, 1 = very good, 2 = good, 3 = fair, 4 = poor |
| ferritin | Blood test, ferritin | Double [L/mL] |
| crprot | Blood test, c-reactive protein | Double [mg/L] |
| hemoglobin | Blood test, hemoglobin | Double [g/dL] |
| gluc | Blood test, fasting glucose | Double [mg/dL] |
| ldl | Blood test, LDL cholesterol | Double [mg/dL] |
| hdl | Blood test, HDL cholesterol | Double [mg/dL] |
| tot_chol | Blood test, total cholesterol | Double [mg/dL] |
| trig | Blood test, triglycerides | Double [mg/dL] |
| hba1c | Blood test, HbA1c | Double [%] |
| systolic_bp | Systolic blood pressure | Double [mmHg] |
| diastolic_bp | Diastolic blood pressure | Double [mmHg] |
| outcome | Diabetes status at 10 years after the baseline | 0 = no diabetes, 1 = diabetes|

### Dataset splitting
First of all, variables are converted in the appropriate data type: *age, education, depression scale, mod_vig_pa and sr_poor_health*
into integers, since they are ordinal variables, and f erritin into numeric. Before splitting the data into training and
test set, 3 vectors are created in oreder to identify which variables are factors, integers or numbers. Afterwards, a function
to split the data in training set and test set is defined, with the option to sample with or without replacement. In this process,
the stratification has been performed as well: the proportion of the two different levels (0/1) of the varaible outcome has been
maianteined the same. The splitting ratio is 80% training and 20% test.
It’s also worth to mention that the percentage of missing values (NAs) for each variable is almost the same in every dataset
as well. It’s important to notice the high percentage of missing values for the varialbe glu and keep into account that the
imputation of such a high number of datapoints for one variable can bias its true contribution in determining the outcome.

### Colinearity check
To check for the presence of colinearity between variables, the correlarion matrix between them has been calculated. As can
be seen from it, there is a high correlarion value (>0.8) between the variables *ldl* and *tot_chol* as well as for the variables *waist*
and *bmi*. For this reason, one of the variables in each couple can be taken out of the analysis without affecting the results,
since the information they carry is the same.
![f_1](https://github.com/Andre1411/Medical-Big-Data---Diabetes/assets/107708093/8fda066d-bb6f-42aa-b146-6aff5625683c)


### Outlier remotion
For the remotion of the outliers, a visual insepction of the numerical variables has been performed and some datapoints have
been removed for the variables presenting most outstanding outliers: *crprot*, *trig* and *ferritin*. The distributions of these
variables and the datapoints considered outliers have been reported in the following figures. The dataset lceaned of the outliers
has been used as sampling source in all the following bootstrap and cross validation procedures.
![Rplot09](https://github.com/Andre1411/Medical-Big-Data---Diabetes/assets/107708093/e086bed7-f6bd-4612-8237-f9a4e04efe27)
![f_2](https://github.com/Andre1411/Medical-Big-Data---Diabetes/assets/107708093/1f9bac88-6785-4dd0-b935-e3dc31cc4fec)

### Normalization
After that, a function to apply the normalization to the datasets is defined and used on both the test and train sets in order to
bring the values of numerical and ordinal variables between 0 and 1.

### Full model
After the pre processing of the data, the full linear model has been fitted to the training data, using all the variables to predict
the outcome, and the performance has been evaluated on the test data. All the results (model coefficients, ROC plot and AUC
values) are reported in the homonymous section below.

### Bootstrapping
The bootstrap analysis has been carried out to understend which features are selected using the same method multiple times and to compare
two different methods of features selection: recursive feature elimination and lasso regression.
In this processes, the stratified sampling of the internal training set (with replacement), the imputation and the normalization have
been performed at each iteration.
For the lasso regression, a 5-fold cross-validation has been performed to find the optimal value of $\lambda$ as the biggest value achieving an AUC of the ROC
not differing more than 0.005 from the the maximum value of all.
At the end of this procedure, the following values for the maximum and the selected λ respectively have been found:
$\lambda_{max} = 0.0003$ and $\lambda_{sel} = 0.002$.
After the bootstrap feature selection, two final models have been finally run, using the features selected by the rfe and the lasso methods with a frequency
higher than 75%.
![lambda_cv](https://github.com/Andre1411/Medical-Big-Data---Diabetes/assets/107708093/1fd236cd-2376-4b2a-b1da-b24e8d27123a)

## Results

### Full model
![f_3](https://github.com/Andre1411/Medical-Big-Data---Diabetes/assets/107708093/21980ef4-ae8f-4bd5-8b37-6490701b7155)

### Feature selection
| Feature | RFE % | RFE selected | LASSO % | LASSO selected |
| --- | --- | --- | --- | --- |
| gender | 93 | yes |100  |yes |
| age | 90 | yes |0 | no |
| education | 96 | yes | 0 | no |
| smoking | 25 | no | 50 | no |
| bmi | 100 | yes | 100 | yes |
| marital_status | 49 | no | 66 | no |
| depression_scale | 74 | no | 79 | yes |
| mod_vig_pa |57 | no | 0 | no |
| heartd_hx |23 | no | 8 | no |
| hypertension | 54 | no | 97 | yes |
| high_col | 15 | no | 13 | no | 
| sr_poor_health | 79 | yes | 99 | yes
| ferritin | 96 | yes | 92 | yes |
| crprot | 18 | no | 4 | no |
| hemoglobin | 34 | no | 4 |
| gluc | 100 | yes | 91 | yes |
| hdl | 35 | no | 0 | no |
| tot_chol |51 | no | 0 | no |
| trig | 76 | yes | 88 | yes |
| hba1c | 100 | yes | 100 | yes |
| systolic_bp | 97 | yes | 75 | yes |
| diastolic_bp | 40 | no | 5 | no |

### Reduced model: recursive features elimination
![f_4](https://github.com/Andre1411/Medical-Big-Data---Diabetes/assets/107708093/d244c63b-4de4-4120-92bc-5b0ceeb68bbe)

### Reduced model: LASSO regression
![f_5](https://github.com/Andre1411/Medical-Big-Data---Diabetes/assets/107708093/c901dbf2-fbc2-4934-ba96-a705b9e6acd9)



## Final comments and observations
Features selection has proven to be a strong method to detect the most important features contributing to determine different
outcomes. In fact, the predictive capability of the model is not affected by the usage of a lower number of features, as can be
seen from the values of the AUC of the ROC: 0.863 for the full model, 0.859 for the reduced one with features selected with
rfe and 0.863 for the reduced model with features selected with lasso regression bootstrapping.
Comparing the two different methods of features selection, is noticeable that most of the selected features are the same, but
there are some differences as well. In particular, some variables selected by one method but not by the other, like age and
education, are seleted with very high frequency by the RFE method, and never by the lasso method. For this reason, their role
in the prediciton of the outcome may need to be inspected with more detail.
Independently from the feature selection method, looking at the coefficients of the different models, the variable hba1c has
always the biggest (and positive) coefficient, suggesting that this variable has a big impact on the probabilty of developing
diabetes during the 10 years after the baseline.

