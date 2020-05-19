#Bouston House Dataset Step by Step


# 1. Prepare Problem
# a) Load Libraries 
# Load all the required modules and libraries that will be used for our problem.
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error


# b) Load Dataset
# Load tha dataset from file/url and this is the place to reduce the sample of dataset specially if it's too large to work with.
# We can always scale up the well performing models later.

filename= 'housing.csv'
names= ['CRIM', 'ZN','INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', "TAX", 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset= read_csv(filename, delim_whitespace=True,names=names)

# 2. Summarize Data
# This step is to learn more about the dataset which will help us decide which algorithms to use with this data. 
# a) Descriptive Statistics
#Shape
print(dataset.shape)
#types of data
#print(dataset.dtypes)
print(dataset.dtypes)
#head
print(dataset.head())
#description
set_option('precision', 1)
print(dataset.describe())
#correlation
set_option('precision', 2)
print(dataset.corr(method='pearson'))

# b) Data Visualizations
#histigram to better understand the distribution of each attribute
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()
#density plot for clearer graphs
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False, fontsize=1)
pyplot.show()
#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8)
pyplot.show()

#Multidimensional data visualization
#scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

#correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# 3. Prepare Data
# a) Data Cleaning - Removing duplicates, dealing with the missing values
# b) Feature Selection
# c) Data Transforms - Scaling/standarizaion of data
#standardrizion is done later on.


# 4. Evaluate Algorithm
# a) Split-out validation dataset
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
#Bseline Algotihms
# b) Test options and evaluate metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'
# c) Spot Check Algorithms
models = []
models.append(("LR", LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR(gamma='auto')))
#evaluate ALgos
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# d) Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algo Comparsion')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# 5. Improve Accuracy
#different scales of the data could be causing problems with the learning abilities of models, especially for SVR and KNN.
#Standardization could help us so let's try that. In standardized data each attribute has a mean value of 0 and std of 1.

#standardization of dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scalar', StandardScaler()),('LR', LinearRegression())])))

pipelines.append(('ScaledLASSO', Pipeline([('Scalar', StandardScaler()),('LASSO', Lasso())])))

pipelines.append(('ScaledEN', Pipeline([('Scalar', StandardScaler()),('EN', ElasticNet())])))

pipelines.append(('ScaledKNN', Pipeline([('Scalar', StandardScaler()),('KNN', KNeighborsRegressor())])))

pipelines.append(('ScaledCART', Pipeline([('Scalar', StandardScaler()),('CART', DecisionTreeRegressor())])))

pipelines.append(('ScaledSVR', Pipeline([('Scalar', StandardScaler()),('SVR', SVR(gamma='auto'))])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)
#compare with box plot
fig = pyplot.figure()
fig.suptitle('Scaled Algo Comparision')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
#The graph shows that KNN is performing the best in comparision to other Algos, so we will tune KNN further
# a) KNN Algorithm Tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold,iid=True)
grid_result = grid.fit(rescaledX, Y_train)
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip (means, stds, params):
    print('%f (%f) with : %r' % (mean, stdev, param))


# best results are Using k =1 and MSE o -19.53



# b) Ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor(n_estimators=10))])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor(n_estimators=10))])))
results =[]
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)

#compare with box plot
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algo Comparision')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#Tune scaled ET as it is showing the best results
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = ExtraTreesRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, iid=True)
grid_result = grid.fit(rescaledX, Y_train)
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip (means, stds, params):
    print('%f (%f) with : %r' % (mean, stdev, param))
    
    
# We get the best results at n_estimators = 150 with MSE of - 9.058, around 1.6 units better than untuned model.


# 6. Finalize Model
scaler = StandardScaler().fit(X_train)
rescaled = scaler.transform(X_train)
model = ExtraTreesRegressor(random_state=seed, n_estimators=150)
model.fit(rescaledX, Y_train)
# a) Predictions on validation dataset and trandform the val data
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))
