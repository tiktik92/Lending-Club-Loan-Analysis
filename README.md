Capstone Project – Python

Predict whether a customer will default on his/her loan.


Introduction


XYZ Corp. is a lending platform that provides loan to its customers based on certain metrics. The corporation has past data of containing financial standing, credit history, payment information, indicator of default and many more. Based on this past information, the corporation intends to build a model to predict default in the future. This will help the company in deciding whether or not to pass the loan.

Number of observations in the given dataset: 855969

Number of variables in the given dataset: 73


Summary of the data/ Review of Literature


v	We can see that there are a total of 855969 individuals of which 809502 (94.57%) are non-defaulters and 46467 (5.43%) are defaulters. 


v	The loan amount ranges from $500 to $35000. Investors can also invest their wealth by funding part or whole of any of the loan application, though the applicant may not be aware of this fact. 


v	The corporation issues loans in two term periods i.e. 36 months and 60 months starting from the year 2010, up till then only 36 month loans were approved. The loans can be individual or joint. 


v	The corporation has its own grading system which is further divided into sub-grades. It also contains three categories for verification status. The corporation also has the provision of a payment plan where customers present a plan how of they intend to repay the loan, but only 5 customers have any kind of plan.



v	Of the 73 variables, 21 variables have extremely high number of missing values (upto 98.44%). Another set of 5 variables have missing values ranging from 49000 to 235000 and the last set of 5 variables have missing values ranging from 32 to 8862.



v	The corporation does not serve all the states in the US. And certain states are highly loss making states where bad loans outnumber good loans. 


Exploratory Data Analysis
1. There has been a steady year-on-year increase in the loans granted.

2. The above figure confirms the increase in loans. It is also important to note that there has been an increase in defaulters as well. The Year 2015 may not follow the trend, but there is likely hood that customers may default in the future and the number may rise. Thus a dynamic approach to this is required.

3. It is clearly evident that post 2008, there has been a clear distinction between the average annual incomes of defaulters and non-defaulters. This may be a very important factor when it comes to deciding who to grant and who not to grant a loan. But somehow it seems to have gone un-noticed.

4. Certain states, IA and ID are where the corporation has no presence, or does not serve these states. 
States of MT, NC, VT and WY are the ones where there are just defaulters or very-very fewer non-defaulters. Other states like AL, FL and IL are high loss making states. The remaining states have no or fewer defaulters.

5. Incurring debt to overcome debt (debt consolidation and credit card) is where the corporation has most defaulters. And as stated earlier, these numbers could rise. Instead there are other areas which look more promising for the corporation. This fact is true even in terms of employment length.
 
6. A similar kind of observation can be made from the defaulting status, the home ownership of the customer and the average interest rate charged.

 



Interestingly, it is usually a trend that unsecured or unverified loans are costlier, but it seems that average interest rates are higher for verified sourced and comparatively lower for non-verified sources. This could be one of the reasons as to why not only non-verified customers but some verified customers may also default.

  
Data Pre-Processing & Transformation

Treating missing data


The missing data must be appropriately treated to ensure accurate analysis. Missing data can be treated in the following ways:

1.	Discarding variables with missing values greater than 50% 

The 21 variables where there was high percentage of missing data (upto 98%) can be discarded as they contain no vital information and imputing values for this amount of missing data will introduce biasness in the model and the model is sure to not perform well.

data.drop([], axis=1, inplace=True) 

Where [] is the list of variables that need to be discarded.

2.	Missing value imputation

The two sets of 5 variables (10 in all) are the ones where the missing values are a small percentage of the data and its imputation may introduce no bias or very little bias in the model. Whether biasness is introduced and what effect it may have on the model is a trial and error process and running multiple iterations of models can help determine whether to impute or not.

data[x].fillna(mean(data[x]), inplace=True)     --> for numeric data types
data[x].fillna(mode(data[x])[0], inplace=True)     --> for categorical data types

Where ‘x’ is the variable whose missing values are to be imputed.




Data Encoding


Data Encoding is a technique of converting Categorical data type to Numeric. This is essential because most algorithms cannot take non-numeric data as input, and thus this transformation is important. Though Decision Trees and other tree based algorithms can very well handle any kind of data type, the more basic algorithms like linear models or Support Vector Machines are not capable of handling such data types. 
Before Encoding
 

After Encoding
 
Data Standardization


Data standardization is a process in which data attributes within a data model are organized to increase the cohesion of entity types. In other words, the goal of data standardization is to reduce and even eliminate data redundancy, an important consideration for application developers because it is incredibly difficult to store objects in a database that maintains the same information in several places.

Python’s sklearn library has 2 methods for implementing standardization:

1.	StandardScaler: Standardize features by removing the mean and scaling to unit variance.

2.	MinMaxScaler: Transforms features by scaling each feature to a given range. This estimator scales and translates each feature individually such that it is in the given range on the training set, i.e. between zero and one. This transformation is often used as an alternative to zero mean, unit variance scaling


Dimensionality Reduction


Dimension Reduction refers to the process of converting a set of data having vast dimensions into data with lesser dimensions ensuring that it conveys similar information concisely. These techniques are typically used while solving machine learning problems to obtain better features for a classification or regression task.


Principal Component Analysis (PCA): In this technique, variables are transformed into a new set of variables, which are linear combination of original variables. These new set of variables are known as principle components. They are obtained in such a way that first principle component accounts for most of the possible variation of original data after which each succeeding component has the highest possible variance.


Steps performed

1.	Installing and importing the necessary packages:
i.	Pandas 
ii.	Matplotlib
iii.	Seaborn
iv.	Numpy
v.	Scikit-learn

2.	Loading the data in Python
i.	XYZCorp_LendingData.txt is a tab (\t) separated file which is read into Python using Pandas’ read_csv() method.

3.	Inspecting the data
i.	Checking the data types of the variables.
ii.	Checking the variable with missing values and the number of missing values in them.

4.	Discarding the variables with missing values greater than 50%

5.	Creating a split list
i.	Creating a datetime series of the ‘issue_d’ variable.
ii.	Creating a Boolean list for date values greater than Jun 2015 to split the data into train and test.

6.	Handling Missing values
i.	Checking for variables with less than 50% missing values.
ii.	Checking which variable is of numeric type and which is of non-numeric type.
iii.	Imputing numeric variables with their respecting mean values and non-numeric variables with their respective mode values.

7.	Discarding variables with no intrinsic meaning and information

8.	Selecting important variables
i.	Selecting variables which are important for the domain and the problem in hand and sub-setting the dataset accordingly.

9.	Segmenting the dataset
i.	Segmenting the dataset into Independent and Dependent variables.
ii.	‘default_ind’ is the dependent variable and the remaining are independent variables.

10.	Label Encoding
i.	Encoding/converting the Categorical data types to Numeric data types

11.	Scaling the data using sklearn’s StandardScaler() method

12.	Performing Dimensionality Reduction
i.	Performing Principle Component Analysis to retain 99.9% of the variation in the data
ii.	This leads to reduction in the number of independent variables from 31 to 27. 
iii.	Transforming the data as per PCA.

13.	Dividing the dataset into training data and testing data based on the splitting list created earlier

14.	Model Building 
i.	Model 1: Fitting a simple Logistic Regression model which gives probabilities of an individual being a defaulter or not. Threshold value of 0.5 is used to divide the negative and the positive class.
ii.	Model 2: A decision tree classifier is used for to determine the outcome of the dependent class.
iii.	Model 3: A bagging technique of RandomForestClassifier is used to improve the performance of the above models. It also uses decision trees but in a bagging approach to overcome the disadvantages that a decision tree has.

15.	Model Evaluation
i.	Confusion Matrix for all the above models was created.
ii.	Models were evaluated on three different evaluation metrics: Precision, Recall and Area Under the Curve (AUC).


Important Definitions:

1.	Precision: Precision is the ratio of correctly predicted positive values to the total predicted positive values. This metric highlights the correct positive predictions out of all the positive predictions. High precision indicates low false positive rate.

                                             

2.	Recall (Sensitivity): The recall is the ratio of correctly predicted positive values to the actual positive values. Recall highlights the sensitivity of the algorithm i.e. out of all the actual positives how many were caught by the program.

                                        

3.	F1 Score: It is the weighted average of Precision and Recall. At first glance, F1 might appear complicated.  It is a much more sophisticated metric than accuracy because it takes both false positives and false negatives into account. Accuracy is suitable only when both false positives and false negatives have similar cost (which is quite unlikely).

      

Results
Model 1: LogisticRegression
 
Model 2: DecisionTreeClassifier
 
Model 3: RandomForestClassifier – Threshold of 0.5

Model 4: RandomForestClassifier – Threshold of 0.6
 


Final Notes

It is clearly evident from the confusion matrices and evaluation metrics for models 1 and 2 that they are not suitable due to very high Type 1 errors. Models 3 and 4 modelled using Random Forest are very close in terms of the evaluation criteria and most suitable. Out of the two models, Model 3 is most appropriate for the following reasons:

Considering average loan amount of $14745 and an average interest rate of 13.20%

1.	In Model 3 the corporation would make a potential loss of a maximum of $943680 on defaulting loans and a potential loss of business of $58980 yielding returns worth $7785 annually.

2.	In case of Model 4, corporation would make a potential loss of a maximum of $884700 on defaulting loans and a potential loss of business of $412860 yielding returns worth $54498 annually.

3.	So, Model 4 would save $58980 but risk losing $46713 annually. Whereas, Model 3 would risk losing $58980 more, but in turn give returns worth $46713 annually.

Thus Model 3 would be the preferred model for XYZ Corp. 


References

1.	https://www.newgenapps.com/blog/precision-vs-recall-accuracy-paradox-machine-learning

2.	https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall

3.	http://scikit-learn.org/stable/modules/classes.html

4.	Images.google.com
