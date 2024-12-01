# from load_csv import load
# from sys import exit
# import pandas as pd
# import matplotlib.pyplot as plt


def train(file: pd.DataFrame):
	fig, ax = plt.subplots()
	ax.scatter(file['km'], file['price'])
	ax.set_ylabel("Price")
	ax.set_xlabel("km")
	mean_y = sum(file['price']) / len(file['price'])
	plt.axhline(y=mean_y, color='r', linestyle='--')
	plt.show()


import numpy as np 
import matplotlib.pyplot as plt 
from load_csv import load

# def estimate_coef(x, y): 
#     n = np.size(x) 
#     m_x, m_y = np.mean(x), np.mean(y) 

#     SS_xy = np.sum(y*x) - n*m_y*m_x 
#     SS_xx = np.sum(x*x) - n*m_x*m_x 

#     theta_1 = SS_xy / SS_xx 
#     theta_0 = m_y - theta_1*m_x 

#     return(theta_0, theta_1) 

# def plot_regression_line(x, y, theta): 
# 	x_array = x.to_numpy()
# 	y_array = y.to_numpy()
# 	plt.scatter(x_array, y_array, color = "b",marker = "o", s = 30) 
# 	y_pred = theta[0] + theta[1]* x_array
# 	plt.plot(x_array, y_pred, color = "r") 
# 	plt.xlabel('price')
# 	plt.ylabel('kms')
# 	plt.show() 


def main():
	try:
		file = load("data.csv")
		# x = file['km']
		# y = file['price']
		# theta = estimate_coef(x, y) 
		# print("Estimated coefficients:\ntheta_0 = {} \ntheta_1 = {}".format(theta[0], theta[1])) 
		# data = open("data", "x")
		# data.write(str(theta))
		# plot_regression_line(x, y, theta) 
	except FileNotFoundError:
		print("File Not Found")
		exit(1)
	train(file)


if __name__ == "__main__":
	main()

	# y = mx + c
	# theta0 = c
	# theta1 = m

	# y = (theta1 * x) + theta0
	# theta0 = (y - y_mean) - theta1 * (x - x_mean) 
	# theta1 = SSxy / SSxx 
	# SSxy = sum(x * y) - (n * (x - x_mean) * (y - y_mean))
	# SSxx = sum(x*x) - n * (x_mean * x_mean)