import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

years = np.array([2000, 2002, 2005, 2007, 2010])
percentages = np.array([6.5, 7.0, 7.4, 8.2, 9.0])
years_norm = (years - years.mean()) / years.std()

def gradient_descent(x, y, learning_rate, iterations):
    m, b = 0, 0
    n = len(x)

    for _ in range(iterations):
        y_predicted = m * x + b
        dm = (-2 / n) * sum(x * (y - y_predicted))
        db = (-2 / n) * sum(y - y_predicted)
        m -= learning_rate * dm
        b -= learning_rate * db

    return m, b

learning_rate = 0.01
iterations = 1000
m, b = gradient_descent(years_norm, percentages, learning_rate, iterations)

year_12 = (12 - b) / m
year_12 = year_12 * years.std() + years.mean()

print(f"Model regresji liniowej: y = {m:.3f} * x + {b:.3f}")
print(f"Procent bezrobotnych przekroczy 12% w roku: {int(np.round(year_12))}")

fig, ax = plt.subplots()
line, = ax.plot(years, percentages, "ro")

def animate(i):
    m, b = gradient_descent(years_norm, percentages, learning_rate, i)
    y = m * years_norm + b
    line.set_ydata(y)
    return line,

ani = FuncAnimation(fig, animate, frames=iterations, interval=10, blit=True)
plt.xlabel("Lata")
plt.ylabel("Procent")
plt.title("Regresja liniowa z wykorzystaniem spadku gradientu")
plt.show()