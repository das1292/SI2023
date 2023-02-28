import numpy as np

# Zadanie 2

# wczytanie danych
data = np.genfromtxt('dane/_info-data-discrete.txt', dtype='str')

# pobranie nazwy systemu
system_name = data[0][0]

# pobranie liczby atrybutów i liczby obiektów
num_attributes = int(data[0][1])
num_objects = int(data[0][2])

# wczytanie typów atrybutów
attribute_types = np.genfromtxt('dane/' + system_name + '-type.txt', dtype='str')

# pobranie indeksów atrybutów numerycznych
numeric_attributes = np.where(attribute_types == 'n')[0]

# pobranie indeksów atrybutów symbolicznych
symbolic_attributes = np.where(attribute_types == 's')[0]

# Zadanie 3

# a) wypisanie istniejących w systemie symbole klas decyzyjnych
classes = np.unique(data[:, -1])
print('Symbole klas decyzyjnych:', classes)

# b) wielkości klas decyzyjnych (liczby obiektów w klasach)
class_counts = []
for c in classes:
    count = np.sum(data[:, -1] == c)
    class_counts.append(count)
print('Liczby obiektów w klasach:', class_counts)

# c) minimalne i maksymalne wartości poszczególnych atrybutów (dotyczy atrybutów numerycznych)
min_values = np.min(data[:, numeric_attributes].astype(np.float), axis=0)
max_values = np.max(data[:, numeric_attributes].astype(np.float), axis=0)
for i in range(len(numeric_attributes)):
    print('Minimalna wartość dla atrybutu', numeric_attributes[i], ':', min_values[i])
    print('Maksymalna wartość dla atrybutu', numeric_attributes[i], ':', max_values[i])

# d) dla każdego atrybutu wypisanie liczby różnych dostępnych wartości
unique_value_counts = []
for i in range(num_attributes):
    if i in numeric_attributes:
        unique_value_counts.append(1)
    else:
        unique_values = np.unique(data[:, i])
        unique_value_counts.append(len(unique_values))
    print('Liczba różnych wartości dla atrybutu', i, ':', unique_value_counts[i])

# e) dla każdego atrybutu wypisanie listy wszystkich różnych dostępnych wartości
for i in range(num_attributes):
    if i in numeric_attributes:
        print('Atrybut', i, 'jest numeryczny')
    else:
        unique_values = np.unique(data[:, i])
        print('Atrybut', i, 'różne wartości:', unique_values)

# f) odchylenie standardowe dla poszczególnych atrybutów w całym systemie i w klasach decyzyjnych (dotyczy atrybutów numerycznych)
data_numeric = data[:, numeric_attributes].astype(np.float)
std_devs = np.std(data_numeric, axis=0)
print('Odchylenie standardowe dla poszczególnych atrybutów w całym systemie:', std_devs)
for c in np.unique(data[:, -1]):
    data_numeric_class = data_numeric[data[:, -1] == c]
    std_devs_class = np.std(data_numeric_class, axis=0)
    print('Odchylenie standardowe dla poszczególnych atrybutów w klasie', c, ':', std_devs_class)

