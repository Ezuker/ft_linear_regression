from parse import search
from sys import argv, exit


def predict(theta0: float, theta1: float, mileage: float):
	estimatePrice = (theta1 * mileage) + theta0
	print(f"The estimated price is equal to: {estimatePrice}$")

def main():
	theta0 = 0
	theta1 = 0
	if len(argv) != 2:
		print("You need to specify a float in argument")
		exit(1)
	try:
		file = open("data", mode = 'r')
		lines = str(file.readlines())
		theta0, theta1 = tuple(search("({:f}, {:f})", lines))
		float(argv[1])
	except TypeError as e:
		print("Data file is wrong")
		exit(1)
	except FileNotFoundError as e:
		print("Data not found, theta0 and theta1 are equals to 0")
	except ValueError as e:
		print("You need to specify a float in argument")
		exit(1)
	predict(theta0, theta1, float(argv[1]))



if __name__ == "__main__":
	main()