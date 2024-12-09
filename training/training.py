from load_csv import load
from sys import exit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d


def grad_descent(X, y, theta, learning_rate, n_iterations):
	"""
	Calcul the best param of theta0 and theta1

	a = a - alpha * (∂J / ∂a)
	b = b - alpha * (∂J / ∂b)
	"""
	cost_history = np.zeros(n_iterations)
	for i in range(0, n_iterations):
		theta = theta - learning_rate * grad(X, y, theta)
		cost_history[i] = cost(X, y, theta)
	return theta, cost_history


def grad(X, y, theta):
	"""
	Function that calculate ∂J(a,b) / ∂a and ∂J(a,b) / ∂b
	"""
	m = len(y)
	return 1 / m * X.T.dot(model(X, theta) - y)


def cost(X, y, theta):
	"""
	MSE Formula
	Erreur quadratique moyenne
	"""
	m = len(y)
	return 1 / (2 * m) * np.sum((model(X,theta) - y) ** 2)


def model(X, theta):
	"""
	Multiplication de matrices
	x1 1
	x2 1   x   a
	xn 1       b
	"""
	return X.dot(theta)


def R_Squared(y, prediction):
	"""
	Calculate the accuracy of the model with the R2 technique
	"""
	return 1 - (sum((y - prediction) ** 2)) / (sum((y - y.mean()) ** 2))


def plt_normalize(x_scaled, y_scaled, X, theta_final):
	plt.subplot(2, 2, 1)
	plt.scatter(x_scaled, y_scaled)
	prediction = model(X, theta_final)
	plt.title("Normalized data")
	plt.plot(x_scaled, prediction, c='r')


def plt_cost(cost_history):
	plt.subplot(2, 2, 2)
	plt.plot(cost_history)
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.title('Cost vs Iterations')


def plt_denormalize(x, y, theta_final):
	plt.subplot(2, 2, 3)
	plt.scatter(x, y)
	plt.title("Denormalized data")
	new_theta_final = np.zeros((2, 1))
	deltaX = max(x) - min(x)
	deltaY = max(y) - min(y)
	new_theta_final[0] = deltaY * theta_final[0] / deltaX 
	new_theta_final[1] = (deltaY * theta_final[1]) + min(y) - theta_final[0] * (deltaY / deltaX) * min(x)  # a
	new_X = np.hstack((x, np.ones(x.shape)))
	prediction2 = model(new_X, new_theta_final)
	print(f"The precision of the algorithm is equal to {R_Squared(y, prediction2)}")
	plt.plot(x, prediction2, c='r')
	return new_theta_final


def plt_3d(X, y_scaled):
	plt.subplot(2, 2, 4, projection='3d')
	theta0 = np.arange(-4, 4, 0.05, dtype=float)
	theta1 = np.arange(-4, 4, 0.05, dtype=float)

	Theta0, Theta1 = np.meshgrid(theta0, theta1)

	cost_history = np.zeros_like(Theta0)
	for i in range(Theta0.shape[0]):
		for j in range(Theta1.shape[1]):
			theta = np.zeros((2, 1))
			theta[0] = Theta0[i, j]
			theta[1] = Theta1[i, j]
			cost_history[i, j] = cost(X, y_scaled, theta)
	ax = plt.gca()
	ax.plot_surface(Theta0, Theta1, cost_history, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
					alpha=0.3)	
	ax.contour(Theta0, Theta1, cost_history, zdir='z', offset=-2.5, cmap='coolwarm')

	ax.set(xlim=(-4, 4), ylim=(-4, 4), zlim=(-5, 20),
		xlabel='Theta0', ylabel='Theta1', zlabel='CostHistory')
	plt.show()
	pass


def main():
	try:
		file = load("data.csv")
		x = np.array(file['km']).reshape(-1, 1)
		y = np.array(file['price']).reshape(-1, 1)
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

		plt.figure(figsize=(12, 8))
		plt_normalize(x_scaled, y_scaled, X, theta_final)
		plt_cost(cost_history)
		new_theta_final = plt_denormalize(x, y, theta_final)
		plt_3d(X, y_scaled)
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