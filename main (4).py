import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm

################ PART 1 ################
#Perform exploratory data analysis and feature selection 

# Load the data into a pandas dataframe
df = pd.read_csv('AirQualityUCI.csv', thousands='.', decimal=',')

# Data cleaning

#drop NA values 
df = df.replace(-200, np.nan)
for col in df.columns:
  df = df.dropna(subset = [col])

# exploratory data analysis (EDA)
print(df.head())     
print(df.shape)            
print(df.info())           
print(df.describe())

df = df.drop(["Date", "Time"], axis=1)

#drop outliers using z-score method

z_threshold = 3

# Loop through each column in the DataFrame
for col in df.columns:
    # Calculate the Z-score for the column
    z_scores = stats.zscore(df[col])
    outliers = (abs(z_scores) > z_threshold)
    df_clean = df[~outliers]

#dataframe after data cleaning
print(df_clean.head())

#feature selection using SelectKBest algorithm
X = df_clean.drop("C6H6(GT)", axis=1)
y = df_clean["C6H6(GT)"]

k = 10  # number of features to select
selector = SelectKBest(score_func=f_regression, k=k)
X_new = selector.fit_transform(X, y)
feature_names = X.columns[selector.get_support()]

df_selected = pd.concat([df_clean[feature_names], y], axis=1)

#dataframe after feature selection
print(df_selected.head())

print("\n Selected Features: \n")
for col in df_selected.columns:
  print(col)
  
################ PART 2 ################
# Visualization

df_selected.plot(kind='box', subplots=True, layout=(4,4), figsize=(15,10), title='Boxplot for each column')

df_selected.hist(figsize=(15,10))

corr = df_selected.corr()
sns.heatmap(corr, annot=True, cmap="YlGnBu")

print(corr)
plt.show()

################ PART 3 ################
#PCA

features = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T"]

#separate data into x and y
x = df_selected.loc[:, features].values
y = df_selected.loc[:, 'C6H6(GT)'].values

#Standardize the data
x = StandardScaler().fit_transform(x)

#Use pca
pca = PCA(n_components=2)
xPrime = pca.fit_transform(x)

#concatenate the data
reducedDimensions = pd.DataFrame(data = xPrime, columns = ['Components Affecting Benzene', 'Benzene'])
reducedDimensionsPlot = pd.concat([reducedDimensions, df_selected[['C6H6(GT)']]], axis = 1)
print("PCA completed\n")

#plot the 2d data
plt.scatter(reducedDimensions['Components Affecting Benzene'], reducedDimensions['Benzene'], c=df_selected['C6H6(GT)'], cmap='viridis')
plt.xlabel('Components Affecting Benzene')
plt.ylabel('Benzene')
plt.title('PCA Plot of Reduced Dimensions and Benzene')
plt.colorbar()
plt.show()

################ PART 4 ################
#K-fold Cross Validation
k = 5
num_validation_samples = len(reducedDimensionsPlot) // k

np.random.shuffle(reducedDimensionsPlot)

validation_scores = []
for fold in range(k):
    validation_data = reducedDimensionsPlot[num_validation_samples * fold: num_validation_samples * (fold + 1)]
    training_data = reducedDimensionsPlot[:num_validation_samples * fold] + reducedDimensionsPlot[num_validation_samples * (fold + 1):]
    model = svm()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)

validation_score = np.average(validation_scores)

print(validation_score)


model = get_model()
model.train(test_data)
test_score = model.evaluate(validation_data)

################ PART 5 ################
#Training the Model

# specify the response variable
response_var = 'C6H6(GT)'

# create the feature matrix and response vector
X = df.drop(response_var, axis=1)
y = df[response_var]

# create the Gaussian Naive Bayes model
nb = GaussianNB()

# fit the model to the data
nb.fit(X, y)

#Lasso Regression
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1

# create a sequential model
model = Sequential()

# add a dense layer with L1 regularization and 10 units
model.add(Dense(10, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l1(0.001)))

# add an output layer with one unit
model.add(Dense(1, activation='linear'))

# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Reshape the data to a column vector
x = x.reshape(-1, 1)

# Create a polynomial features object with degree 2
poly = PolynomialFeatures(degree=2)

# Transform the input data to include polynomial features
x_poly = poly.fit_transform(x)

# Fit a linear regression model to the transformed data
model = LinearRegression()
model.fit(x_poly, y)

# Evaluate the model on some new data
x_new = np.linspace(-15, 15, num=100).reshape(-1, 1)
x_new_poly = poly.transform(x_new)
y_new = model.predict(x_new_poly)

# Plot the original data and the model predictions
plt.scatter(x, y)
plt.plot(x_new, y_new, color='red')
plt.show()

#SVM Regression
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense

# create an SVM regressor
svr = SVR(kernel='linear', C=1.0, epsilon=0.1)

# fit the regressor to the training data
svr.fit(X_train, y_train)

# create a sequential model
model = Sequential()

# add a dense layer with one unit and no activation function
model.add(Dense(1, input_shape=(X_train.shape[1],), activation=None))

# compile the model with the hinge loss function and the SVM regressor as the optimizer
model.compile(loss='hinge', optimizer=svr)

# evaluate the model on the testing data
score = model.evaluate(X_test, y_test)
