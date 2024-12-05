from load_csv import load
from sys import exit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def grad_descent(X, y, theta, learning_rate, n_iterations):
	cost_history = np.zeros(n_iterations)
	for i in range(0, n_iterations):
		theta = theta - learning_rate * grad(X, y, theta)
		cost_history[i] = cost(X, y, theta)
	return theta, cost_history


def grad(X, y, theta):
	m = len(y)
	return 1 / m * X.T.dot(model(X, theta) - y)


def cost(X, y, theta):
	m = len(y)
	return 1 / (2 * m) * np.sum((model(X,theta) - y) ** 2)


def model(X, theta):
	return X.dot(theta)


def main():
	try:
		file = load("data.csv")
		x = np.array(file['km'])
		y = np.array(file['price'])
		x = x.reshape(-1, 1)
		y = y.reshape(-1, 1)
		theta = np.zeros((2, 1))
		
		# Standardization
		# mean_x = np.mean(x)
		# std_x = np.std(x)
		# x_scaled = (x - mean_x) / std_x

		# Normalization
		x_scaled = (x - min(x)) / (max(x) - min(x))
		y_scaled = (y - min(y)) / (max(y) - min(y))

		# Gradient descent with scaled x
		X = np.hstack((x_scaled, np.ones(x.shape)))
		theta_final, cost_history = grad_descent(X, y_scaled, theta, 0.01, 10000)

		plt.figure(figsize=(12, 5))

		plt.subplot(2, 2, 1)
		plt.scatter(x_scaled, y_scaled)
		prediction = model(X, theta_final)
		plt.title("Normalized data")
		plt.plot(x_scaled, prediction, c='r')

		plt.subplot(2, 2, 2)
		plt.plot(cost_history)
		plt.xlabel('Iterations')
		plt.ylabel('Cost')
		plt.title('Cost vs Iterations')

		plt.subplot(2, 2, 3)
		plt.scatter(x, y)
		new_theta_final = np.zeros((2, 1))
		deltaX = max(x) - min(x)
		deltaY = max(y) - min(y)
		new_theta_final[0] = deltaY * theta_final[0] / deltaX 
		new_theta_final[1] = (deltaY * theta_final[1]) + min(y) - theta_final[0] * (deltaY / deltaX) * min(x)  # a
		print(f"expected new_theta_final (8481.172796984529, -0.020129886654102203)")
		new_X = np.hstack((x, np.ones(x.shape)))
		prediction2 = model(new_X, new_theta_final)
		plt.plot(x, prediction2, c='r')
		plt.show()
	except FileNotFoundError:
		print("File Not Found")
		exit(1)
	try:
		data = open("data", "w")
		tuple_value = (float(new_theta_final[0][0]), float(new_theta_final[1][0]))
		data.write(str(tuple_value))
		data.close()
	except:
		pass


if __name__ == "__main__":
	main()

#Equation normal (nul)
	# y = mx + c
	# theta0 = c
	# theta1 = m

	# y = (theta1 * x) + theta0
	# theta0 = (y - y_mean) - theta1 * (x - x_mean) 
	# theta1 = SSxy / SSxx 
	# SSxy = sum(x * y) - (n * (x - x_mean) * (y - y_mean))
	# SSxx = sum(x*x) - n * (x_mean * x_mean)