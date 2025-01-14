# from scipy.optimize import linprog

# # Objective function (minimize x + y)
# c = [3, 2]  # Coefficients for x and y

# # Constraints in the form of Ax <= b
# A = [[-5, -4], [-2, -3], [-1, -1]]  # Transformed constraints
# b = [-40, -24, -8]  # Right-hand side values for constraints

# # Bounds for the variables (x >= 0, y >= 0)
# bounds = [(0, None), (0, None)]

# # Solve using linprog
# res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

# # Check the result
# if res.success:
#     print(f"Optimal Solution: {res.x}")
#     print(f"Optimal Value: {res.fun}")
# else:
#     print(f"Optimization failed: {res.message}")

import numpy as np
import matplotlib.pyplot as plt

# Define the constraints (example: x + y <= 6 and x - y >= 2)
def constraint1(x):
    return 6 - x

def constraint2(x):
    return x - 2

# Create a range of x values
x = np.linspace(0, 10, 400)

# Calculate the corresponding y values for each constraint
y1 = constraint1(x)
y2 = constraint2(x)

# Plot the constraints
plt.plot(x, y1, label=r'$x + y \leq 6$')
plt.plot(x, y2, label=r'$x - y \geq 2$', linestyle='--')

# Fill the feasible region
y_feasible = np.minimum(y1, y2)
plt.fill_between(x, y_feasible, where=(y_feasible > 0), color='gray', alpha=0.5, label='Feasible Region')

# Set axis limits
plt.xlim(0, 10)
plt.ylim(0, 10)

# Labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.grid(True)
plt.legend()

plt.show()