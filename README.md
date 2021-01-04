# Data Mining Project 2020/2021
# Data Understanding and Preparation
1. **Data Understanding**: explore the dataset with the analytical tools
	studied and write a concise “data understanding” report describing data
	semantics, assessing data quality, the distribution of the variables and the
	pairwise correlations.
	* Data semantics
	* Distribution of the variables and statistics
	* Assessing data quality (missing values, outliers)
	* Variables transformations & generation
	* Pairwise correlations and eventual elimination of redundant variables
2. **Data Preparation**: improve the quality of your data and prepare it by extracting new features interesting for describing the customer profile and his purchasing behavior.
# Clustering analysis
Based on the customer’s profile explore the dataset using various clustering techniques.
Carefully describe your decisions for each algorithm and which are the advantages provided by the different approaches.
1. Clustering Analysis by K-means
	* Identification of the best value of k
	*  Characterization of the obtained clusters by using both analysis of the k centroids and comparison of the distribution of variables within the clusters and that in the whole dataset
	* Evaluation of the clustering results
2. Analysis by density-based clustering :
	* Study of the clustering parameters
	* Characterization and interpretation of the obtained clusters
3. Analysis by hierarchical clustering
	* Compare different clustering results got by using different version of the algorithm
	* Show and discuss different dendrograms using different algorithms
4. Alternative clustering techniques in the library: [pyclustering](https://github.com/annoviko/pyclustering/)
	* Fuzzy C-Means
	* Genetic Algorithms
5. Final evaluation of the best clustering approach and comparison of the clustering obtained

# Classification Analysis
Consider the problem of predicting for each customer a label that defines if (s)he is a **high-spending** customer, **medium-spending** customer or **low-spending** customer.
1. Define a customer profile that enables the above customer classification. Please, reason on the suitability of the customer profile, defined for the clustering analysis. In case this profile is not suitable for the above prediction problem you can also change the indicators.
	* KMeans clustering labels
	* Fuzzy C-Means clustering labels
2. Perform the predictive analysis comparing the performance of different models discussing the results and discussing the possible preprocessing that they applied to the data for managing possible problems identified that can make the prediction hard. Note that the evaluation should be performed on both training and test set.
	* Decision Tree
	* Random Forest
	* Naïve Bayes
	* KNN
	* SVM
	* Neural Networks
	* Oversampling using SMOTE
3. Final analysis on model explainability

# Sequential Pattern Mining
1. Consider the problem of mining frequent sequential patterns. To address the task:
	* Model the customer as a sequence of baskets
	* Apply the Sequential Pattern Mining algorithm ([gsp](DM_10_TASK4/gsp.py) implementation)
	* Discuss the resulting patterns
2. Handling _time constraint_ while building Sequential Patterns
