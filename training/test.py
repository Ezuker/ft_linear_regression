import numpy as np
import matplotlib.pyplot as plt

# Define the slope (a) and intercept (b)
a = 2  # Slope of the line
b = 1  # Intercept of the line

# Generate x values (for example, from -10 to 10)
x = np.linspace(-10, 10, 100)

# Calculate the corresponding y values using y = ax + b
y = a * x + b

# Plot the line
plt.plot(x, y, label=f"y = {a}x + {b}", color="blue")

# Add labels and title
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Plot of y = ax + b")

# Add a legend
plt.legend()

# Display the plot
plt.grid(True)  # Optional, adds grid lines
plt.show()