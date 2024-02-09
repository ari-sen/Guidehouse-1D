# Predicting Ukraine's Emerging Humanitarian Needs, Guidehouse-1D
## Fall 2023, AI Studio Project Write-Up

### Forecasting Ukraine displacement, utilizing K-Means Clustering and SARIMAX Time Series :

**Table of Contents :**

1. [Business Focus](https://github.com/ari-sen/Guidehouse-1D/blob/main/README.md#business-focus)
2. [Data Preparation and Validation](https://github.com/ari-sen/Guidehouse-1D/blob/main/README.md#data-preparation-and-validation)
3. [Modelling](https://github.com/ari-sen/Guidehouse-1D/blob/main/README.md#modelling)
4. [Results](https://github.com/ari-sen/Guidehouse-1D/blob/main/README.md#result--achieved-an-average-rmse-of004149-across-4-clusters-25-regions-implying-high-accuracy)
5. [Evaluation & Looking Forward](https://github.com/ari-sen/Guidehouse-1D/blob/main/README.md#evaluation--looking-forward)
6. [Credits](https://github.com/ari-sen/Guidehouse-1D/blob/main/README.md#credits)
### Business Focus 


This project is a response to the humanitarian crisis in Ukraine, which was invaded by Russia in February 2022. The crisis has resulted in tens of thousands of people impacted through casualties and internal displacement. More than 8.2 million civilians fled the country, which by April 2023 created Europe’s largest refugee crisis. Besides the mentioned impacts, there was major environmental damage caused by the war which contributed to food crises worldwide and food aid. The ML along with our time series model aims to use the ACAPS Ukraine Master dataset to forecast patterns such as certain needs and the demographics of people internally displaced within different regions known as Oblasts. Additionally, the primary goal of our model besides forecasting needs, was to get any valuable insight that would benefit and guide humanitarian efforts and organizations that aim to provide aid to those affected. Our results for our time series model was an RMSE score of 0.04149 which indicated high accuracy in predicting registered IDPs.


**Keep in mind that through this project we primarly wanted to:**

- Utilize data analysis and forecasting techniques, by using AI/ML principles, to draw predictions towards future humanitarian efforts required
- Acknowledge the sensitivity of the humanitarian crisis in Ukraine within our data manipulation process.
- Solutions from our ML model(s) to be adequate to Guidehouse Advanced Analytics & Intelligent Automation (AAIA) team’s future clientele.

### EXPLORATORY DATA ANALYSIS
We divided exploratory data analysis between each other as we had 526 columns of data. Firstly, looking at the data before finding any patterns or trends, we noticed that most of the outliers were of low values, that being 0 fatalities which was accounted for by the month the data was recorded. Most of the outliers occurred in January 2022 - February 2022, which made sense since that was the beginning of the invasion. 

To get an overview of which features had strong or weak correlations, we first created a heat map.
[![image]] https://github.com/ari-sen/Guidehouse-1D

Since we had a lot of features, we anticipated the heatmap to be big. As we observed, there’s a pattern of 1’s, a duplicate correlation of each feature, such as a correlation between battles and battles. Despite that, many features like people in level 3 humanitarian conditions and registered IDPs had a strong correlation of .61, and km2 of Russian forces consistently had lower correlations with IDPs. This gave us great insight into what to expect for our cluster analysis. 

Next, we experimented with inertia and distortion to apply the elbow method to find the optimal number of clusters. For distortion, we found that the number of optimal clusters was 5 and inertia resulted in 4. In the end, we decided to expand upon the inertia method and apply PCA to 4 cluster groups using scalar standardization. Before applying PCA, we had to convert the most relevant categorical feature to a numerical value: Oblasts and scale the time column. For this dilemma, there were two methods: giving each Oblast a number for our time series model or creating dummies for the Oblasts. In the end, thanks to Aaleia, we created dummy variables for the regions which eased our PCA process as we were able to include Oblasts in our analysis and not lose much information geographically. The heat map and k-means analysis guided us to further evaluate the clusters in depth.


### Data Preparation and Validation
#### DATASET DESCRIPTION

We were originally given 6 datasets provided by the humanitarian data exchange as linked: Ukraine 2023 Humanitarian Needs Overview People in Need, Ukraine Response Activities, Ukraine Flash Appeal, and Humanitarian Needs Overview (2021  - 2023), and ACAPS Ukraine Master Dataset. Some datasets included a variety of categorical and quantitative features such as Oblasts, # of people in need, severity score, amount of people exposed, and number of fatalities while others had the amount of people that received specific humanitarian aid like education.


Since the master dataset had all of the features in the remaining ones, combined we decided to pre-process just the master dataset1. We combined both master datasets from 2022 and 2023, removing completely null columns, containing total data for Ukraine or Crimea, as much of Crimea’s data was missing. This was the perfect dataset choice as we weren’t able to combine the other 6 datasets for the features to be aligned with one another and it was representative of all 24 regions of Ukraine, unlike the others that didn’t contain the cities. The master dataset was also diverse in demographics and features that would represent the severity of the crisis:
 - Internally displaced people - contextually meaning anyone who has been forced to leave their home as a way to avoid armed conflict but still reside within the Ukrainian borders
 - Humanitarian Condition Level ( 1- 5) - originates from the European’s INFORM Severity Index and is based on:
 - Impact of crisis
 - People in need
 - Condition of people
 - Access to humanitarian needs - healthcare, education, shelter, food


Terminology:

**IDPs** – Internally Displaced Person/People, contextually meaning anyone who has been forced to leave their home as a result to avoid armed conflict in this case but still reside within the Ukranian borders.

**Humanitarian Condition Level** - based on the European’s INFORM Severity index which determines a level 1-5, based on:
- Impact of crisis
- People in need
- Condition of people
  
**Access to humanitarian needs** - Healthcare, shelter, food, etc.

`# male population` and `# total older population (60 years and up)` provided a stark constrast in the data, since Ukraine restricted men from the ages 18-60 from leaving the borders in case of a need for fighters. 

The `# registered IDPs` feature was used as a label for the supervised time series model.

We chose this particular dataset because its dataset size was bigger than others that were provided through HDX, had features necessary for our approach, and was representative of all of the oblasts and their categorical features.


#### *ACAPS MASTER DATASET* PREPROCESSING

Before preprocessing we made the following adjustments :

- Removed all non-numerical values (e.g. postal code), except for Oblast (region)​
- Removed data on wages, income, pension, and inflation​ since it was not relevant
- Did not use food/fuel cost data, as it was too likely to be affected by other factors​
- Removed data not available for every Oblast​
- Removed data from totality countries features (Crimea and Ukraine)​

And we used the following methods : 

**Time Scaling** - Implemented a Time column, based on days since Jan 2022 using the monthly data, in preparation for the time series model where we forecast future IDPs.

**Standardization** - Standard Scaler for all numerical features through Z-score normalization (all eligible columns except Oblast)
Standard scaler: mean value 0 and standard deviation 1 (removing the mean and and scaling to unit variance) for better PCA centroid performance to handle variance within the same scale.

**Dummy/Indicator Variables** - Converting each Oblast column to numerical by giving each oblast a number for time series.

In total, our master dataset had 34 features, but we decided to use 32 out of 34 features since it would be representative of reasons why internally displaced people numbers may be higher than other data points. Feature importance was particularly important for our model as we wanted to account for as much inclusivity as we could when addressing the crisis. 

We made sure that our model focused on the following features:

Area controlled by Ukraine and Russia

Population and people exposed

Demographics

Levels of Severity from 1-5 (Minimal, stressed, moderate, severe, extreme)

Violence

Fatalities

Unemployment

Since we wanted to start with the unsupervised learning approach first to find more broad patterns in our data, this was relevant to understanding the territorial dynamics through geographical context. It also aligned with our goal to create an unbiased model since we were including demographics of age and gender, it allowed for our model to capture how the crisis impacted different sectors of the population. Lastly, since we didn’t include fuel costs, unemployment was an alternative to representing the correlation between any economic instability and displacement. 

![image](https://github.com/Aaleia/Guidehouse-1D/assets/143746727/275e7177-0156-4dde-8101-332f9c7a623a)


### Modelling

Since there were many features that correlated to each geographical situation within each oblast, we wanted to observe broader patterns within our dataset through clustering, and then further use them for a fixed supervised learning numerical prediction for IDPs. 

#### Unsupervised Learning : K-Means Clustering 

K-Means Clustering gathers data points that are similar to each other in some shape or form, and groups them into each clusters, and measures the data points based upon the sum of the squared distances between each point and the mean of its assigned cluster. The way that unsupervised learning works is that the model is supposed to derive the clusters based on patterns it observes in the data, without being explicitly told of prior expected labels for each predictive pattern observation segmenting regions for targeted aid towards the features that will end up within their own cluster.

Visualized through PCA (Principal Component Analysis), we used the elbow method to find the optimal value of the hyperparameter `K`, as `K = numbers of clusters`. The elbow method finds where the rate of decrease sharply changes within the plotted cluster, minimizing the total variance within each cluster. The optimal number was 4 different clusters, .

<img width="568" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/11a1e7fd-1859-4717-ac43-fc0e0f85bf7e">

<img width="403" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/cc9a7801-cc80-4b38-80e4-4fa4171eed5a">

#### Individual Cluster Analysis
Each cluster and the data points within them was expressed through the individual oblasts it was tied to.

<img width="913" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/57690e64-5baa-450c-a01a-fae007f94745">


**Cluster 0:** Cherkaska, Chernihivska, Chernivetska, Ivano-Frankivska, Khmelnytska, Kirovohradska, Mykolaivska, Poltavska, Rivnenska, Sumska, Ternopilska, Vinnytska, Volynska, Zakarpatska, Zhytomyrska

**Cluster 1:** Khersonska, Luhanska, Zaporizka

**Cluster 2:** Dnipropetrovska, Kharkivska, Kyiv, Kyivska, Lvivska, Odeska

**Cluster 3:** Donetska

**Donetska has its own cluster reserved for itself, because while cluster 2 contains Kyiv, the captial of Ukraine, Donetska holds the largest population with the steepest decrease in IDPs. According to geographical data driven from the dataset, Russia has focused most of its efforts within this region, due to its proximity to the border.** 

We utilized a heatmap correlation matrix for each cluster, in order to find a broader pattern between IDPs and the categorical features it related to. **Red** represents a very strong correlation, **Gray** is neutral, and **Blue** represents a very weak correlation. `# female older population (60 years and up)`, `# people affected`, `# km^2 unconfirmed control`, `# km^2 controlled by Ukrainian authorities/forces`, `# male population`, and `# total older population (60 years and up)` had the most correlation to `# registered IDPs`overall. `# km^2 controlled by Russian forces` was used as a control variable in order to compared the two controlled regions against each other when it came to displacement.

<img width="649" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/a7bd3407-3ee3-412d-b06c-dd5c0697f013">  

Majority of the male and older female populations are correlated highly with registered IDPs, but less with impacted regions in Ukraine and Russia (Russia is somewhat correlated to IDPs compared to Ukraine)

<img width="591" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/1e2674dd-58aa-4049-95f3-aeba223431f6">

Majority of unconfirmed and confirmed regions of Ukraine are correlated with registered IDPs, but less or none with the majority of the male and older female populations. (Russia does not correlate with IDPs at all)

<img width="626" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/e640b6f2-cd71-4428-8205-4d9aed5b9354">
<img width="619" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/694aaecc-7186-44ee-8d1f-c4a2760aeb64">

People affected are highly correlated with registered IDPs, but less with impacted regions in Ukraine and Russia and the majority of the male and older female populations (Cluster 1 takes more in account for older populations, while Cluster 0 takes more in account of unconfirmed control of land)

#### **Km^2 of Russian Forces consistently has lower correlations for IDPs.**

#### Supervised Learning : Seasonal ARIMAX Time Series

Seasonality is defined within our dataset as external circumstances that changes the time series data in a certain shape or form, during different monthly time intervals.

Seasonal ARIMAX (Autoregressive Integrated Moving Average with eXogenous variables) is a historically fit time series data model based on ARIMA, with the use of seasonal patterns in order to forecast trends within phases of time. The autoregressive aspects implements the relationship between a current observation and a specified number of lagged observations, making the time series data itself stationary for differencing, models the relationship between an observation and a residual error from a moving average model, and accounts for eXogenous variables that not included specifcially in the time series data, but may alter the results. 

In our case, our supervised model finds trends or systemic patterns over the period of 2 years within our *ACAPS Master Dataset* assigned labels , which produces successive measurements over fixed time intervals, and aligns well with our monthly data. In fact, it performed better for certain oblasts than just ARIMAX, performed well even when seasonality wasn’t present, and was not likely to be affected by weather, more likely patterns such as June 2022 and June 2023 (buildup before Ukrainian counteroffensives).

**TRAINING + TESTING :** 

Training data was used to predict IDPs in each oblast. The hyperparameter `M` had a value of `12`, as `M = yearly seasonality` from our monthly data. In order to split our training and testing data accordingly, we used an 85-15 split to include certain spikes in IDP data (also performs well with an 80-20 split). We also applied Grid Search to find optimal order and seasonal order for each oblast, and to avoid overfitting.
 
  -> Was optimized by using K-Means clusters that grouped the oblasts together, so we only ran grid search 4 times instead of 25 times, while utilizing our broader patterns that we observed for consistentcy during the evaulation of the SARIMAX model.

<img width="1181" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/eb61b6cc-e94f-4de8-8d9a-3f71ade04d9a">

## **Result : Achieved an average RMSE of 0.04149 across 4 clusters (25 regions), implying high accuracy**

<img width="999" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/727faf7a-bbaf-4f71-b1d3-dd1e5b099094">

#### Note : The time scale listed on the x-axis is the default set year value from the 1970s and is representative of the range of 2022-2023.

RMSE (Root Mean Squared Error) is calculated as the square root of the average of the squared differences between the predicted and actual values. The regression predicion error of our model is significantly low, suggesting that the time series data is properly fit to the model, and that the model is able to forecast an approximate amount of IDPs per region, on unseen data. The graph is attained from Cluster one (Kyiv) which shows the test set and predictions linearly increase, which may be due to the test set's emphasis on `# people affected` and `# total older population (60 years and up)` feature for that cluster. 

### Evaluation & Looking Forward

Humanitarian needs is tricky to predict in a militarized/politicized conflict.

With the geopoltical landscape changing rapidly for a variety of different reasons, it would not be viable or ethical to proclaim these findings to a government official for a complete use case, as we are quite limited in scope as to what we can throughly predict, given our reduced computational resources and external measures not recorded or altered in our dataset. However, within what we were able to accomplish with our smaller machine learning models, if you have a fine tuned datset with broader patterns to each categorical feature, then predicting displacement is possible., and thus enhance aid distribution planning, reducing the crisis's human impact.
 
As for ethical concerns, we thought that having a fixed prediction number alone was not sufficient for this problem, so we opted towards clustering in order to get a better hold of the relationships between IDPs and their categoricla features. For future research, we would need to incorporate more real time data, as well as responsibly increasing our computational resources needed for wider scale models. 

### Credits

Aaleia - Completed the SARIMAX time series model and the K-Means Clustering model.

Sherin - Completed cluster analysis and business understanding of the problem.

Cindy - Choosing between supervised and unsupervised learning.

Arielle - Chose dataset and preprocessed the data.

Iman - Researched ethics, ML papers, and evaluated our problem for future use cases.

Thank you to Break Through Tech, the Cornell Tech AI Program team, and Guidehouse for granting us this learning opportunity that we will never forget! To our Challenge Advisor, Karen, and teammates, thank you for all of your support and contributions.
