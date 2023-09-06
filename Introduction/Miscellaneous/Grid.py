import matplotlib.pyplot as plt
import numpy as np

lst = [[10, 11, 12], [20, 21, 22]]
ls = [[10, 11, 12], [10, 11, 12]]




for key, value in zip(*lst):
    print (key, value)

print("~~~~~~~~~~~~~~~")

for key, value in zip(*ls):
    print (key, value)

for item in zip(*ls):
    print (item)

transpose = list(map(list, zip(*ls)))
print(transpose)

# for key, value in zip(lst):
#     print (key, value)

print(zip(*lst))




# Generate a range of x values
x = np.linspace(-10, 10, 100)

# Define the function we want to plot
y = x**2

# Create the plot
plt.figure(figsize=(12.19, 6.86))  # Create a new figure
plt.plot(x, y)  # Plot y = x^2

# Set labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = x^2')

# Add grid lines
plt.grid(True)

# Display the plot
plt.show()

