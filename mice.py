from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.model_selection import train_test_split
import numpy as np

class MiceImputer(object):
	def __init__(self, missing_values=np.nan, strategy="mean", verbose=0, copy=True):
		self.missing_values = missing_values
		self.strategy = strategy
		self.verbose = verbose
		self.copy = copy
		self.imp = SimpleImputer(missing_values=self.missing_values, 
                                strategy=self.strategy,
                                verbose=self.verbose, copy=self.copy)

	def _seed_values(self, X):
		np.random.seed(123)
		# self.imp.fit(X)
		return self.imp.fit_transform(X)
			
	def _process(self, X, column, model_class,X_tmp,**k):
		# Remove values that are in mask
		mask = np.array(self._get_mask(X, self.missing_values)[:, column].T)[0]
		mask_indices = np.where(mask==True)[0]
		# print("++++",mask)
		X_data = np.delete(X, mask_indices, 0)
		# Instantiate the model
		model = model_class(**k)

		# Slice out the column to predict and delete the column.
		y_data = X[:, column]
		X_data = np.delete(X, column, 1)
		X_temp = np.delete(X_tmp, column, 1)

		# print("y_data : ", y_data)
		# Split training and test data
		# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)

		train_indices = np.where(~np.isnan(y_data))[0]
		test_indices = np.where(np.isnan(y_data))[0]
		
		# split X_data and y_data into train and test sets
		X_train_tmp = X_temp[train_indices] 	# full data 
		y_train_tmp = y_data[train_indices] # k NaN <- target
		X_predict = X_temp[test_indices]
		# print('=====<> ', X_predict)
		y_test = y_data[test_indices] # NaN ?
		# Fit the model
		# print(X_train_tmp.shape,y_train_tmp.shape, '----->', X_predict.shape )
		model.fit(X_train_tmp, y_train_tmp)

		# Score the model

		# Predict missing vars
		y_predict = model.predict(X_predict)

		# Replace values in X with their predictions	
		X[test_indices, column] = y_predict
		X_tmp[test_indices, column] = y_predict
		# Return model and scores
		return X,X_tmp
	
	def _get_mask(self,X, value_to_mask):
		if value_to_mask == "NaN" or np.isnan(value_to_mask):
			return np.isnan(X)
		else:
			return X == value_to_mask

	def transform(self, X, model_class=BayesianRidge, n_iter=10):
		index_of_nan = np.where(np.isnan(X).any(axis=0))[0]
		X = np.matrix(X)
		mask = self._get_mask(X, self.missing_values)
		seeded = self._seed_values(X)
		# specs = np.zeros((iterations, len(X.T)))
		# c = np.sum(np.isnan(X).any(axis=0))
		# print(index_of_nan)
		for i in index_of_nan:
			# specs[i][c] = self._process(X, c, model_class)
			# print(i)
			X,seeded = self._process(X, i, model_class,seeded)
		
		# Return X matrix with imputed values
		return X