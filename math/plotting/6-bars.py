#!/usr/bin/env python3
"""
Plot a stacked bar graph
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    This function creates a stacked bar chart to represent
    the quantity of different fruits owned by three people:
    Farrah, Fred, and Felicia.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    people = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']

    bottom = np.zeros(3)
    bar_width = 0.5

    for i, row in enumerate(fruit):
        plt.bar(people, row, color=colors[i], label=fruits[i],
                bottom=bottom, width=0.5)
        bottom += row

    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))
    plt.legend()
    plt.show()
