# How I created type 2 diabetes classifiers on NHANES data using R 

![image](https://user-images.githubusercontent.com/75398560/123784120-74fc4900-d91a-11eb-8e86-389c47994f47.png) [_image source_](https://www.mz-store.com/blog/diabetes-and-physical-exercise-contraindications-a-post-training-meal-for-a-diabetics/)

### Background on Type 2 Diabetes
Diabetes is the fastest growing chronic disease in Australia, with one individual developing this disease every five minutes (Diabetes Australia, 2020). Additionally, diabetes contributes to one in every ten deaths in Australia (Australian Institute of Health and Welfare [AIHW], 2020). Furthermore, more than 90% of cases with type 2 diabetes (T2D) may be _avoided_ if accompanying preventable risk factors such as unhealthy eating habits and a sedentary lifestyle are modified (Willett, 2002).

In Australia alone, estimated costs are $14.6 billion dollars for T2D annually (Lee et al., 2012). Shockingly, in Australia for every five people who are diagnosed, _four are left undiagnosed_ (Valentine et al., 2011). Therefore, detecting and predicting disease onset in individuals is the first step to prevention and management of T2D progression. I aim to evaluate machine learning classification models of LASSO logistic regression, random forest, Naïve Bayes and XGBoost to detect and accurately classify patients with T2D. 

## Data :page_facing_up:
The National Health and Nutrition Examination Survey [NHANES](https://wwwn.cdc.gov/nchs/nhanes/default.aspx) is a continuous biennial survey program beginning from 1999, used to assess the mental health, physical health, and nutrition of American people (Centre for Disease Control and Prevention in America [CDC]). 

On average the survey collects a nationally representative random sample of 5,000 responses per year. This novel study will utilise more than 100,000 observations between 1999-2018 and 12,300 variables. I used the first four categories of NHANES datas as similar questions were asked in the ‘Examination’ and ‘Questionnaire’ categories to the ‘Dietary’ category. 

1.	Demographics
2.	Examination
3.	Laboratory
4.	Questionnaire 
5.	Dietary

NHANES dataset incorporates medical and physiological data (e.g., BMI, blood pressure mm/Hg), as well as laboratory data (e.g., LDL cholesterol readings mg/dL). 

This in combination with demographic, dietary, and health-related behavioural questions (e.g., alcohol consumption in past 12 months) enable researchers to contextualise major disease prevalence including relevant risk factors and therefore, NHANES data is widely used in healthcare research (Dong et al., 2019;  McDowell et al., 2004; Zhu et al, 2020). 

The data was retrieved from the `nhanesA` package in R and analysed in R. As the aim of the study is to create a classification model to predict T2D, participants were identified as diabetic if they answered “yes” to the question “_Have you been told by a doctor you have diabetes?_” Participants who answered “no” were identified as non-diabetic. 

## 1. Libraries :books:
Good coding practice is keeping all your libraries in a chunk and commenting what each library does. I used the following libraries for this project. 
```r
library(nhanesA) #importing NHANES data
library(dplyr) #manipulating data
library(purrr) #manipulating data
library(finalfit) #initial data analysis
library(caret) #ML workflows
library(randomForest)#Random Forest
library(naivebayes) #Naivebayes
library(glmnet)# LASSO regression
library(ggplot2)#visualisation
library(mice) #univariate imputation
library(NADIA) #reusing same univariate imputation on test data
library(plyr) #XGBoost
library(xgboost)#XGBoost
```

## 2. Pre-processing
Researchers in the past studies chose 14 variables including demographics like age and ethnicity, as well as examination components like BMI and hypertension. I included these variables, however, also added variables relating to laboratory data as strong associations to T2D have previously been shown in literature (Akinsegun et al., 2014; Park et al., 2021; Taheri et al., 2018). This resulted in extracting **75** variables during initial feature selection.

### Extracting data
Example shortened code for extracting relevant sub-section data for 2003-2004 Survey. This was done for each survey from 1999-2018. This took me a very long time though its always important to **know your data** :eyes:.
```r
####Extracting relevant sub-section data for 2003-2004 Survey####

#Demographic section
D = nhanes('DEMO_C') #Demographic data

#Examination section
BP = nhanes('BPX_C') #Blood pressure data
BM = nhanes('BMX_C') #Body measures data

#Laboratory section
AL = nhanes('L16_C') # Albumin - Urine
LD = nhanes ('L13AM_C') #LDL Cholesterol
HD = nhanes ('L13_C') #HDL Cholesterol

####Merging all sub-sections together based on sequence number of the survey####
survey_2003_2004 =
  list(D,BP,BM,AL,LD,HD,GL,PL,PRO,BL,ALQ,DI,DB,KI,MC,PH,SM, BPQ,CH) %>% 
  purrr::reduce(full_join, by = "SEQN") 

###Selecting relevant questions from sub-sections###  75 variables
survey_2003_2004 <- survey_2003_2004 %>% 
  dplyr::select(RIAGENDR, RIDAGEYR, RIDRETH1, DMDBORN, DMDYRSUS, DMDEDUC2,INDHHINC, BPXSY1, BPXDI1, BMXWT, BMXHT, BMXBMI, BMXARMC, BMXARML, BMXLEG, BMXWAIST, LBDLDL,
  XSCH, LBXSGL, LBXSATSI,  LBXSASSI, LBXSC3SI,  LBXSGTSI, LBXSIR, LBDSTPSI, LBXSTR, LBXSUA) %>%
  dplyr::rename(
    Gender = RIAGENDR,
    Age = RIDAGEYR,
    Ethnicity = RIDRETH1,
    Country_Birth = DMDBORN,
    Time_US = DMDYRSUS,
    Education_level = DMDEDUC2,
    Income = INDHHINC,
    BP_Sys = BPXSY1,
    BP_Dia = BPXDI1,
    Weight = BMXWT,
    Fast_Glu = LBXGLU,
    Insulin = LBXIN,
    Alcohol = ALQ120Q,
    Diabetes = DIQ010,
    )
 ```

### Merging datasets
Variables were coded inconsistently per survey cycle and were manually renamed. Making sure each variable was coded the same for each survey, I then merged all datasets into one calling it `combined`. Merging all survey cycles resulted in a dataset containing 101,316 observations.
```r
#merging surveys together
combined <- do.call("rbind", list(survey_1999_2000,survey_2001_2002,survey_2003_2004,
                     survey_2005_2006,survey_2007_2008,survey_2009_2010,
                     survey_2011_2012,survey_2013_2014,survey_2015_2016,
                     survey_2017_2018))
```

### Removing 25% missing data
Variables with more than 25% missing were removed as anymore lead to bias in results (Zhuchkova & Rotmistrov, 2021). 
```r
#showing percent of missing values in each column
print(colMeans(is.na(combined))*100)

#removing columns (variables) with more than 25% missingness (25 variables remain)
combined25 = combined[, which(colMeans(!is.na(combined)) >= 0.75)]

#making sure variables are coded correctly
combined25 %>% 
  ff_glimpse()
```

### Data type transformations
25 variables remained and were transformed into numeric or categorical datatypes. Example code:
```r
#converting continuous factors to numeric
combined25$Age <- as.numeric(as.character(combined25$Age))
combined25$Weight <- as.numeric(as.character(combined25$Weight))

#converting categorical factors to factors
combined25$Gender <- as.factor(as.character(combined25$Gender))
combined25$Ethnicity <- as.factor(as.character(combined25$Ethnicity))
```


Levels in variables were renamed to meaningful ones and those levels that contained responses like ‘refused’ or ‘unsure’ were dropped as it added no additional meaning to the data. Example code:
```r
##### - Transforming categorical data levels into meaningful labels - #####
levels(combined25$Country_Birth) <- list("USA" = "1", "Mexico" = "2","Other Country" = c("4", "5"), "Delete" = c("7","9","(Missing)"))

#dropping 'refused','don't know' and other meaningless responses in variables
is.na(combined25$Country_Birth) <- combined25$Country_Birth == "Delete" 
combined25$Country_Birth <- droplevels(combined25$Country_Birth)
```

### Splitting data into train and test
I set seed and split the dataset to train (80%) and test (20%). Setting seed allows you to reproduce your reslts. Click [here](https://www.kaggle.com/obrienmitch94/importance-of-setting-seed-in-model-fitting) for information and examples about setting seed.
```r
set.seed(2021) #setting seed

#80% training, 20% test (stratified random sample)
training_sub <- combined25$Diabetes %>% 
  createDataPartition(p = 0.8, list = FALSE)

train_dat  <- combined25[training_sub, ]
test_dat <- combined25[-training_sub, ]
```

### Single imputation
The data was missing at random [MAR](https://www.theanalysisfactor.com/missing-data-mechanism/), so the probability of a missing observation could be explained by the observed data. I performed single imputation from the `mice` package in R was  on remaining missingness, as missing data may lead to biased outcomes if not handled accordingly (Fan et al., 2014). To prevent information leakage, I imputed test data separately, based on the same training imputation model. This imputation uses predictive mean matching for numeric variables, logit for binary variables, multinomial logit for nominal >2 levels and ordered logit for ordered var >2 levels. 
I used 5 gibbs sampling iterations.
```r
#(5 gibbs sampling iterations)
D_imp_train <- mice(train_dat,m=1, maxit=5, printFlag = FALSE, seed = 2021)
D_imp_trainC <- mice::complete(D_imp_train)

#imputing test data based on the same training imputation model  - prevents information leakage
imp_test <- NADIA::mice.reuse(D_imp_train, test_dat, maxit = 1,printFlag = FALSE, seed = 2021)
imp_testC <- imp_test[[1]]
```

### Downsampling
The diabetes dependent variable in the training data is imbalanced (i.e., _Diabetic_=6420: _Non-Diabetic_=74,633) and was downsampled to the minority class, recommended for large datasets and greater generalisability (Duchesney et al., 2011; Xue & Hall, 2015).
```r
#original training table = >1:10 class imbalanace
set.seed(2021)
downsamp_train <- downSample(x = D_imp_trainC[,-1],
                    y = D_imp_trainC$Diabetes,
                    yname="Diabetes")
```

### Dummy coding, normalisation and standardisation
Categorical variables in the train and test data were dummy coded and all variables were centered and scaled. To prevent information leakage, calculations from training set were used to standardise variables in the test set. This resulted in 31 variables in both sets. 
```r
###DUMMY CODING - dummy code on  train and test data separately  - prevents information leakage
#dummy code  IVs 
D_Xtrain <- model.matrix(Diabetes~., downsamp_train)[,-1]
D_Xtest <- model.matrix(Diabetes~., imp_testC)[,-1]

#numeric conversion of DV - 
D_Ytrain <- ifelse(downsamp_train$Diabetes == "Diabetic", 1, 0)
D_Ytrain <- as.factor(as.character(D_Ytrain))
#performance metrics: balanced accuray, P/R, F1

###STANDARDISATION
#standardisng variables in training set 
D_vals_preproc <- caret::preProcess(D_Xtrain, method = c("center", "scale"))
D_scaledDummy_Xtrain <- predict(D_vals_preproc, D_Xtrain)

#using calculations from training set to standardise variables in the test set  - prevents information leakage
D_scaledDummy_Xtest <- predict(D_vals_preproc, D_Xtest)
```

## 3. Feature Selection
Final feature selection was conducted using LASSO logistic regression as it avoids overfitting the model and helps achieve better classification accuracy (Ludwig et al., 2015). Ten-fold cross-validation was performed whilst parameter tuning the lambda value in LASSO to avoid information leakage. Twenty-six variables were kept in both sets.
```r
#using 10 fold CV for finding best lambda&model fit
set.seed(2021)
fit_lassologit <- cv.glmnet(x=D_scaledDummy_Xtrain, y = D_Ytrain, family = "binomial", alpha = 1, nfolds = 10, standardize = FALSE)

#plot CV error of log lambda, in graph log of optimal lambda is approx
plot(fit_lassologit, xlab="Log Lambda", ylab = "Cross validation error")
#26 variables included in lowest CV error (not including intercept)

#optimal lamba value
fit_lassologit$lambda.min
```
```r
#showing 26 important variables
coef(fit_lassologit, s = fit_lassologit$lambda.min)
```
![image](https://user-images.githubusercontent.com/75398560/123777429-89891300-d913-11eb-8e6d-bc5e58814abb.png)


## 4. LASSO Logistic Regression
The data was overall imbalanced therefore balanced accuracy, sensitivity, and specificity were used (Duchesney et al., 2011; Xue & Hall, 2015). Using LASSO logistic regression an overall balanced accuracy of 83.156%. 
```r
#extracting predictions and class probabilities
prob_lassolog <- predict(fit_lassologit, newx = D_scaledDummy_Xtest, s = "lambda.min", type = "response") #probabilities
pred_lassolog <- ifelse(prob_lassolog >0.5, "Diabetic", "Non-Diabetic") #predicted values

D_actual <- imp_testC$Diabetes #actual values

#confusion matrix
confusionMatrix((factor(pred_lassolog)),(factor(D_actual)))
```
![image](https://user-images.githubusercontent.com/75398560/123776995-2303f500-d913-11eb-932c-2c0fe15b78b5.png)


## 5. Random Forest Model
Five-fold cross-validation was performed during model trainings on training set, with parameter tuning utilised via grid-search for optimal parameters for each model. During validation, models were used with testing data to output predictions for corresponding diabetes class labels. An overall balanced acccuracy of 83.156% resulted. 
```r
control <- trainControl(method = 'cv', number = 5, search = 'grid', allowParallel = FALSE)

#creating data frame with possible tuning values for mtry parameters
tuning_grid <- expand.grid(.mtry=c(4,6,7,8))

#training with various ntrees
set.seed(2021)
rf_fit <- train(x = RD_scaledDummy_XtrainM,
                y = D_Ytrain,
                method = "rf", 
                metric = 'Accuracy', 
                tuneGrid = tuning_grid, 
                trControl = control,
                ntree = 500)


rf_fit$finalModel

#best mtry = 5
```
```r
set.seed(2021)

#extracting predictions and class probabilities
pred_rf <- caret::predict.train(object=rf_fit, RD_scaledDummy_XtestM, type = "raw") #predictions
pred_rf2 <- ifelse(pred_rf == "1", "Diabetic", "Non-Diabetic") #predicted values

D_actual <- imp_testC$Diabetes #actual values

#confusion-matrix

confusionMatrix(factor(pred_rf2),factor(D_actual))
```
![image](https://user-images.githubusercontent.com/75398560/123777918-fc928980-d913-11eb-851f-5aa901082b68.png)


## 6. Naive Bayes Model
Balanced accuracy of 77.122%.
```r
#Parameters to tune
# laplace (Laplace Correction)
# usekernel (Distribution Type)
# adjust (bandwidth adjustment)


#creating 5 fold CV  using gridsearch 
controlnb <- trainControl(method = 'cv', number = 5, search = 'grid', allowParallel=FALSE)

#creating data frame with possible tuning values for  parameters
tuning_gridnb <- expand.grid(laplace=c(0,0.5,1), usekernel=c(TRUE,FALSE), adjust=c(0.5, 1, 1.5,3.0))

#training with various ntrees
set.seed(2021)
nb_fit <- train(x = RD_scaledDummy_XtrainM,
                y = D_Ytrain, 
                method = "naive_bayes", 
                metric = 'Accuracy', 
                tuneGrid = tuning_gridnb, 
                trControl = controlnb,
                usepoisson = TRUE)

```
```r
set.seed(2021)
#extracting predictions and class probabilities
pred_nb <- caret::predict.train(object=nb_fit, RD_scaledDummy_XtestM, type = "raw") #predictions
pred_nb2 <- ifelse(pred_nb == "1", "Diabetic", "Non-Diabetic") #predicted values

D_actual <- imp_testC$Diabetes #actual values

#confusion-matrix

confusionMatrix(factor(pred_nb2),factor(D_actual))
```
![image](https://user-images.githubusercontent.com/75398560/123778210-395e8080-d914-11eb-8226-b61b389c720e.png)


## 7. eXtreme Gradient Boost (XGBoost) Model
83.156% balanced accuracy. 
```r
#Parameters to tune
# nrounds ( Boosting Iterations)
# max_depth (Max Tree Depth)
# eta (Shrinkage)
# gamma (Minimum Loss Reduction)
# colsample_bytree (Subsample Ratio of Columns)
# min_child_weight (Minimum Sum of Instance Weight)
# subsample (Subsample Percentage)

#creating 5 fold CV  using gridsearch
controlxgb <- trainControl(method = 'cv', number = 5, search = 'grid', allowParallel = FALSE)

#creating data frame with possible tuning values for parameters
tuning_gridxgb <- expand.grid(
  nrounds = seq(from = 200, to = 1000, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = seq(from=2, to=24, by=2),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

#training with various ntrees
set.seed(2021)
xgb_fit <- train(x = RD_scaledDummy_XtrainM,
                y = D_Ytrain, 
                method = 'xgbTree', 
                metric = 'Accuracy', 
                tuneGrid = tuning_gridxgb, 
                trControl = controlxgb)
```
```r
#extracting predictions and class probabilities
pred_xgb <- caret::predict.train(object=xgb_fit, RD_scaledDummy_XtestM, type = "raw") #predictions
pred_xgb2 <- ifelse(pred_rf == "1", "Diabetic", "Non-Diabetic") #predicted values

D_actual <- imp_testC$Diabetes #actual values

#confusion-matrix
confusionMatrix(factor(pred_xgb2),factor(D_actual))
```
![image](https://user-images.githubusercontent.com/75398560/123778684-b12cab00-d914-11eb-9219-ba00bd6fbb21.png)


_**NOTE:**_ I'm not perfect and neither is my code :stuck_out_tongue_winking_eye:. I'm learning new/more efficient ways to code all the time, so if you find a better way of doing things then go for that! I'm just putting this code and project out there for those interested in the data science field and to show you what I love doing :grin:.

## References
Australian Institute of Health and Welfare [AIHW]. (2020). Diabetes. https://www.aihw.gov.au/reports/diabetes/diabetes/contents/deaths-from-diabetes.

Centre for Disease Control and Prevention in America [CDC]. (2021). National Health and Nutrition Examination Survey. http://www.cdc.gov/nhanes. 

Diabetes Australia. (2020). Diabetes in Australia. https://www.diabetesaustralia.com.au/about-diabetes/diabetes-in-australia/. 

Dong, Z., Wang, H., Yu, Y., Li, Y., Naidu, R., & Liu, Y. (2019). Using 2003–2014 U.S. NHANES data to determine the associations between per- and polyfluoroalkyl substances and cholesterol: Trend and implications. Ecotoxicology and Environmental Safety, 173, 461–468. https://doi.org/10.1016/j.ecoenv.2019.02.061

Fan, J., Han, F., & Liu, H. (2014). Challenges of Big Data analysis. National Science Review, 1(2), 293–314. https://doi.org/10.1093/nsr/nwt032

Ludwig, N., Feuerriegel, S., & Neumann, D. (2015). Putting Big Data analytics to work: Feature selection for forecasting electricity prices using the LASSO and random forests. Journal of Decision Systems, 24(1), 19–36. https://doi.org/10.1080/12460125.2015.994290

McDowell, M., Dillon, C., Osterloh, J., Bolger, P., Pellizzari, E., Fernando, R., Montes de Oca, R., Schober, S., Sinks, T., Jones, R., & Mahaffey, K. (2004). Hair Mercury Levels in U.S. Children and Women of Childbearing Age: Reference Range Data from NHANES 1999-2000. Environmental Health Perspectives, 112(11), 1165–1171. https://doi.org/10.1289/ehp.7046

Park, J., Lee, H., Park, J., Jung, D., & Lee, J. (2021). White Blood Cell Count as a Predictor of Incident Type 2 Diabetes Mellitus Among Non-Obese Adults: A Longitudinal 10-Year Analysis of the Korean Genome and Epidemiology Study. Journal of Inflammation Research, 14, 1235–1242. https://doi.org/10.2147/JIR.S300026

Taheri, S., Asim, M., Al Malki, H., Fituri, O., Suthanthiran, M., & August, P. (2018). Intervention using vitamin D for elevated urinary albumin in type 2 diabetes mellitus (IDEAL-2 Study): study protocol for a randomised controlled trial. Trials, 19(1), 230–230. https://doi.org/10.1186/s13063-018-2616-5

Willett W. C. (2002). Balancing lifestyle and genomics research for disease prevention. Science, 296(5568), 695–698. https://doi.org/10.1126/science.1071055

Xue, J.-H., & Hall, P. (2015). Why Does Rebalancing Class-Unbalanced Data Improve AUC for Linear Discriminant Analysis? IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(5), 1109–1112. https://doi.org/10.1109/TPAMI.2014.2359660

Zhu, F., Chen, C., Zhang, Y., Chen, S., Huang, X., Li, J., Wang, Y., Liu, X., Deng, G., & Gao, J. (2020). Elevated blood mercury level has a non-linear association with infertility in U.S. women: Data from the NHANES 2013–2016. Reproductive Toxicology (Elmsford, N.Y.), 91, 53–58. https://doi.org/10.1016/j.reprotox.2019.11.005

Zhuchkova, S., & Rotmistrov, A. (2021). How to choose an approach to handling missing categorical data: (un)expected findings from a simulated statistical experiment. Quality & Quantity. https://doi.org/10.1007/s11135-021-01114-w
