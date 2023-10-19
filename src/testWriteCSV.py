import csv

with open('../Barridos.csv', 'a') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['pepito1', 'pepito2', 'pepito3', 'pepito4', 'pepito5', 'pepito6'])