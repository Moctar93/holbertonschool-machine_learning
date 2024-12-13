#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
    Function to plot 5 different graphs in one figure
    with a 3x2 grid layout.
"""
def all_in_one():

    # Data preparation
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Create the figure and the grid layout
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("All in One", fontsize="x-small")

    # Plot 1: y = x^3
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(np.arange(0, 11), y0, 'r-')
    ax1.set_title("y = x^3", fontsize="x-small")
    ax1.set_xlabel("x", fontsize="x-small")
    ax1.set_ylabel("y", fontsize="x-small")

    # Plot 2: Scatter plot
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.scatter(x1, y1, c='magenta', s=5)
    ax2.set_title("Scatter Plot", fontsize="x-small")
    ax2.set_xlabel("x", fontsize="x-small")
    ax2.set_ylabel("y", fontsize="x-small")

    # Plot 3: Exponential decay
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(x2, y2)
    ax3.set_yscale("log")
    ax3.set_title("Exponential Decay", fontsize="x-small")
    ax3.set_xlabel("Time (years)", fontsize="x-small")
    ax3.set_ylabel("Fraction Remaining", fontsize="x-small")

    # Plot 4: Two Exponential Decays
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(x3, y31, 'r--', label="C-14")
    ax4.plot(x3, y32, 'g-', label="Ra-226")
    ax4.set_title("Two Exponential Decays", fontsize="x-small")
    ax4.set_xlabel("Time (years)", fontsize="x-small")
    ax4.set_ylabel("Fraction Remaining", fontsize="x-small")
    ax4.legend(fontsize="x-small")

    # Plot 5: Histogram
    ax5 = fig.add_subplot(3, 2, (5, 6))
    ax5.hist(student_grades, bins=10, edgecolor="black")
    ax5.set_title("Student Grades", fontsize="x-small")
    ax5.set_xlabel("Grades", fontsize="x-small")
    ax5.set_ylabel("Number of Students", fontsize="x-small")

    # Adjust layout for better fit
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Execute the function
if __name__ == "__main__":
    all_in_one()

