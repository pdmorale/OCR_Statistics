import os, csv


with open(file, encoding="utf8", errors='ignore') as csvFile:
    reader = csv.reader(csvFile)
    headers = next(reader, None)

    sorted_list = sorted(reader, key=lambda row: row[0].lower(), reverse=False)
    # for index, column in enumerate(sorted_list):
    # print(column)