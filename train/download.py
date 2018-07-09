import csv

filename = 'train-annotations-bbox.csv'

with open(filename, "r") as csvfile:
    datareader = csv.reader(csvfile)
    count = 0
    for row in datareader:
        print (row)
        count += 1
        if count > 14:
            break