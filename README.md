# Mathematical Programming Solutions

## Description
This project provides solutions to various mathematical programming problems using different programming techniques and methods. Currently, it includes a **Linear Programming** solver using the **Graphical Method**, which allows users to visualize and solve linear programming problems graphically. Additional mathematical programming solutions will be added over time.

## Features
* **Linear Programming Solver (Graphical Method)**
  * Interactive visualization of feasible region and optimal solution
  * Supports both maximization and minimization problems
  * Handles two-variable linear programming problems
  * Dark-themed, high-resolution plots for better readability
  * Real-time solution updates
* Future updates will include:
  * Simplex Method
  * Transportation Problems
  * Assignment Problems
  * Network Models

## Requirements
* Python 3.8 or higher
* Django 3.x
* NumPy
* SciPy
* Matplotlib

## Quick Start

### Installation

1. Clone the repository to your local machine:
```bash
git clone https://github.com/Abiram116/MathCourse.git
```

2. Navigate to the project directory:
```bash
cd <project-directory>
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Django development server:
```bash
python manage.py runserver
```

5. Access the application at `http://127.0.0.1:8000/`

### Using the Web Interface

1. Go to the homepage at `http://127.0.0.1:8000/`
2. Click on "Graphical Solver" in the navigation menu
3. Enter your linear programming problem:
   * Type your objective function (e.g., `3x + 2y`)
   * Add constraints one by one
   * Select maximize or minimize
4. Click "Solve" to see the solution

## API Documentation

### Endpoints

#### 1. Home Page
- **URL**: `/`
- **Method**: `GET`
- **Description**: Landing page with navigation options

#### 2. Graphical Solver Interface
- **URL**: `/graphical-solver/`
- **Method**: `GET`
- **Description**: Interactive interface for solving LP problems

#### 3. Solve Linear Programming
- **URL**: `/solve-linear-programming/`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
    "optimization_type": "maximize",
    "objective": "3x + 2y",
    "constraints": "2x + y <= 8; x + 2y <= 6; x >= 0; y >= 0"
}
```
- **Response**:
```json
{
    "x_value": 3.2,
    "y_value": 1.6,
    "optimal_value": 13.6,
    "img_str": "base64_encoded_image..."
}
```

## Example Problem

### Maximize Profit Example
A company produces two products (X and Y) with following constraints:
* Product X requires 2 hours and Product Y requires 1 hour of labor
* Available labor: 8 hours
* Product X requires 1 hour and Product Y requires 2 hours of machine time
* Available machine time: 6 hours
* Profit: $3 per unit of X and $2 per unit of Y

#### Input:
- Objective Function: `3x + 2y` (maximize)
- Constraints:
  * `2x + y <= 8` (labor constraint)
  * `x + 2y <= 6` (machine time constraint)
  * `x >= 0` (non-negative X)
  * `y >= 0` (non-negative Y)

## Input Format Guidelines

### Objective Function
- Use `x` and `y` as variables
- Coefficients must be numbers
- Terms can be positive or negative
- Example: `3x + 2y` or `-2x + 5y`

### Constraints
- Use `<=` or `>=` for inequalities
- Each constraint on a new line or separated by semicolon
- Must be in the form: `ax + by <= c` or `ax + by >= c`
- Non-negativity constraints (`x >= 0`, `y >= 0`) are automatically added

## Troubleshooting

### Common Issues
* **Invalid Format**: Ensure your input follows the format guidelines
* **No Solution**: Check if your constraints create a feasible region
* **Unbounded Solution**: Verify if your constraints properly bound the feasible region
* **Browser Issues**: Try clearing cache or using a different browser if visualization doesn't load

### Error Messages
* "Invalid JSON data": Check your API request format
* "Invalid constraint format": Verify constraint syntax
* "No feasible solution exists": Constraints are contradictory

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License. See the LICENSE file for details.
