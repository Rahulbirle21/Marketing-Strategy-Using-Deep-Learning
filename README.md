# MARKETING CAMPAIGN STRATEGY USING DEEP LEARNING

---

## __Installation Guide__
1. Clone or Fork the project
2. Create a Virtual Environment and write the given command.
```python
pip install -r requirements.txt
```
---
## __Tools and Libraries used__
* Python
* Jupyter Notebook
* Pandas
* Numpy
* Tensorflow
* Matplotlib
* Seaborn
* Streamlit for Web application

## __Problem Statement__ 
With an increase in market competitiveness and changing customer needs, it has become imperative for the companies to develop a scientific marketing strategy to improve their business outcomes. By using Deep learning with domain knowledge, we can develop an efficient marketing strategy to identify those factors which influence the purchase decisions of the customers and the profit prospects of a company.


## __Introduction Of The Project__
The project aims to develop a Deep learning model for a United Kingdom based company to help them identify the potential buyers of their products in order to minimise the marketing cost to be spent on non buyers and to maximise the profits. For this project, the company has shared 10% of their loyalty programme customer data along with their purchase decisions in the past.Based on the data, each successful buyer generates a revenue of 15,000 Euros for the company, while the marketing and other expenses costs 4000 Euros to the company. This information will be used for Profit prospect analysis along with identification of potential buyers.   

## __Data Collection__
Dataset given by the company : 

The dataset has total 22,223 rows and 10 columns.

#### Attributes of the dataset :

 1. ID : ID to uniquely identify each customer.
 2. DemAffl : Indicates the wealth or financial conditions on a scale of 1 to 30.
 3. DemAge : Age of the customer.
 4. DemClusterGroup : Clusters to segment the customers.
 5. DemGender : Gender (Male,Female and Unknown).
 6. Region : Region or city of residence of the customer.
 7. LoyalClass : Loyalty status (Tin,Silver,Gold,Platinum.
 8. LoyalSpend : Total amount spent by the customer in past.
 9. LoyalTime : Time as loyalty card member.
10. TargetBuy : 1 (indicates buyer) and 0 (indicates non buyer)

---

## __Project Summary__
### 1. __Loading the dataset__
```python
df = pd.read_csv('Project_dataset.csv')
```
---

### 2. __Data Processing And Exploratory Data Analysis__
* Identified the outliers using a boxplot
 
  ![Image Link](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/mboxplot.png)
  
---

* Outlier treatment using the Capping method -- In Capping method, the outlier values are replaced either with the upper limit or with the lower limit of the Interquartile range (IQR)
```python
# Treating the outliers using the Capping method.

for i in df.columns:
    if df[i].dtype!='object':
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        IQR=q3-q1
    
        lower = q1-1.5*IQR
        upper = q3+1.5*IQR
        df[i]=np.where(df[i]>upper,upper,np.where(df[i]<lower,lower,df[i]))
```
---
* Used for loop to fill the null values (Null values in numeric features are replace with mean value of respective columns whereas null values in categorical columns are replaced with mode value of respective columns.)
```python
# Filling the null values

for i in df.columns:
    if df[i].dtype=='object':
        df[i].fillna(df[i].mode()[0],inplace=True)  # Filling the Null values in categorical columns using mode value.
    elif df[i].dtype!='object':
        df[i].fillna(df[i].mean(),inplace=True)    # Filling the null values in numeric columns using mean value.
    else:
        print('Error')
```
---
* Countplot of Loyal Class and insights generated from it.

  ![Image Link](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/loyal%20count.png)
---
* Relationship between Loyal class and Loyalty spending using a Pie chart.
```python
# Relationship between average amount spent by each class using a pie chart

explode = (0.05,0.05,0.05,0.05)
df.groupby('LoyalClass').mean().plot(kind='pie',y='LoyalSpend',autopct='%1.0f%%',explode=explode,legend=False)
plt.axis('off')
plt.show()
```

![Image Link](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/class%20amount.png)

---
* Relationship between Loyalty Spending based on each cluster using pie chart.

![image](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/cluster%20amount.png)
---
---

* Analysis of average affluence grade based on gender

![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/gender%20afflu.png)

---
---
* Relationship between Clusters and Target buyers using stacked bar chart.

  ![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/cluster%20buyers.png)
  * __Observations__:
---
---
* Relationship between loyal class and Target Buyers

 ![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/class%20buyers.png)
---
---
* Distribution of Loyalty spent amount using Histogram

  ![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/amount%20hist.png)
---
---
* Distribution of age of the customers using Histogram.

  ![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/age%20hist.png)
---
* Correlation between numeric features using heatmap

![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/heatmap.png)
---

* Scatterplot of Demographic affluence and amount spent based on Target buy.
  ![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/affl%20buy.png)
  ---

### 3. __Deep Learning (Artifical Neural Network) Model  Building__
* __Ordinal Encoding of Categorical Variables__ : Here, we've used a unique method of ordinal encoding of categorical features based on the 'Credi_limit' variable. For example, a category having higher value of credit limit is given higher order or integer value in the encoding, whereas a category having lower credit limit is given a lower order.
```python
# Ordinal Encoding of categorical features based on the 'Target Buy' variable

df1['DemClusterGroup'] = df1['DemClusterGroup'].replace(['A','B','C','D','E','F','U'],[1,4,6,5,2,3,7])
df1['DemGender'] = df1['DemGender'].replace(['M','F','U'],[2,3,1])
df1['Region'] = df1['Region'].replace(['London','Midlands','Bristol','Cardiff','Birmingham','Wales','Yorkshire',
                                      'Bolton','Nottingham','Leicester','Scotland','Ulster','Trafford'],
                                            [13,12,11,10,9,8,7,6,5,4,3,2,1])

df1['LoyalClass'] = df1['LoyalClass'].replace(['Silver','Tin','Gold','Platinum'],[4,3,2,1]) 
```
* __NOTE__ : Here we've encoded the categorical features based on the output variable , that is, TargetBuy based on the insights generate from EDA. For example, a category having higher number of buyers has been given a higher order or number whereas a category having lower number of buyers has been given a lower order.
---
* __Feature scaling__
```python
# Scaling the features using Standardscaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```
---

### __Deep Learning Model (Artificial Neural Network)  

![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/dL%20model.png)

__Training and Testing loss using line plot__ :

![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/loss.png)

---

___Calculated the threshold Probability using True Positive and False Positive Predictions__ 

```python
fpr, tpr, thresholds = roc_curve(y_test,y_pred)
optimum = np.argmax(tpr-fpr)
opt_threshold = thresholds[optimum]
print('The optimum threshold value for predictions made by the model is', opt_threshold)
```

__The threshold probability values came out to be 0.76__

### __Profit Prospect Analysis__

   1. To analyse the profit prospects, we have grouped all the customers from the test data into 10 decile classes  and sort them based on the Buying Probability predicted by the model on the test data and create a new dataframe 'final_df'.

      ![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/decile.png)
---

   2. Average probability of buying the product based on each decile class.
![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/mean%20prob.png)

---
   3. Profit calculation for each decile class based Revenue and loss data given by the company```python
      # Calculating total profit and loss based on each decile group of customers.
```python      
profit_cal = []
revenue_from_successful_buyer = 15000
promotion_cost = 4000
net_profit = 11000

buyer_list = final_profit_df['Buyers'].tolist()  # converting the dataframe column into list for profit calculation
non_buyers_list = final_profit_df['Non_buyers'].tolist()

for profit,loss in zip(buyer_list,non_buyers_list): 
    total_revenue_from_buyers = (profit * 11000)
    total_loss_from_non_buyers = (loss * 4000)
    profit_cal.append(total_revenue_from_buyers - total_loss_from_non_buyers)
 
# Inserting the Total_profit column to the dataframe
final_profit_df['Total_Profit'] = profit_cal
final_profit_df = final_profit_df.sort_values(by='Total_Profit',ascending=False)
final_profit_df
```
![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/total%20prof.png)

---
---
### __Made Predictions on all the 22,223 customers using our ANN model, and identified each customer as profitable or loss generating customer and given Final Recommendations to the company__

![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/final%20obs.png)

![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/finalll.png)

---
## __Evaluation of the Model__
![im](https://github.com/Rahulbirle21/Marketing-Strategy-Using-Deep-Learning/blob/main/images/model%20evaluation.png)


  ###  __Observation__ : The accuracy of the ANN model came out to be 97% (approx)

---

## __Deployed the Model as a Web application to identify the Potential buyers__

[![Video]()

---
## 6. __Project Highlights__

   1. Easy to use and understand
   2. Open Source
   3. High Accuracy
   4. Deployed as Web Application
   5. Will help the company filter its potential buyers and reduce its marketing cost.
