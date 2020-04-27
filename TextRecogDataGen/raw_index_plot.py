import csv, os
import numpy as np
from difflib import SequenceMatcher
from matplotlib import pyplot as plt


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def generate_graph(filename):
    # noinspection SpellCheckingInspection
    # filename = os.getcwd() + fileN   #-> when exec. from Studio it gives path from .nuget

    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        # for index, column_header in enumerate(header_row):
        # print(index, column_header)

        highs = []

        for row in reader:
            if row[8] == '':
                continue  # There are some empty strings which can’t be converted to int
            high = int(row[8])  # Convert to int
            highs.append(high)  # appending high temperatures

        # print(highs)

        # Plot Data
        fig = plt.figure(dpi=128, figsize=(10, 6))
        plt.plot(highs, c='red')  # Line 1
        # Format Plot
        plt.title("Some stupid OCR vendor performance", fontsize=24)
        plt.xlabel('', fontsize=16)
        plt.ylabel("Performance (Thomas index)", fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.show()
    return  # "done"


def sort_column(file, target):
    # noinspection SpellCheckingInspection

    with open(file) as f:
        reader = csv.reader(f)
        headers = next(reader)
        # for index, column_header in enumerate(headers):
        #    print(index, column_header)

        sorted_list = sorted(reader, key=lambda row: row[headers.index(target)], reverse=False)
        for index, item in enumerate(sorted_list):
            print(index, item)

    return sorted_list


def plot_statistics(file, target):
    with open(file, encoding="utf-8_sig", errors='ignore') as f:
        reader = csv.reader(f)
        headers = next(reader)
        # for index, column_header in enumerate(headers):
        #    print(index, column_header)
        sorted_list = sorted(reader, key=lambda row: row[headers.index(target)], reverse=True)
        last = len(sorted_list[0]) - 1
        # print(os.linesep, "***", os.linesep, sorted_list[0][last])
        # print(len(sorted_list[0]))

        feature, measure = [], []

        for row in sorted_list:
            if row[headers.index(target)] == '':
                continue  # There are some empty strings which can’t be converted to int
            effect = float(row[headers.index(target)])  # Convert to float
            score = float(row[last])
            # print("scored: ", score, "  | blur: ", effect)
            feature.append(effect)  # appending feature
            measure.append(score)

        # calculate polynomial
        z = np.polyfit(feature, measure, 18)
        f = np.poly1d(z)

        # calculate new x's and y's
        x_new = np.linspace(feature[0], feature[-1], 50)
        y_new = f(x_new)

        # Plot Data
        fig = plt.figure(dpi=128, figsize=(10, 6))
        plt.scatter(feature, measure, c='green', label='MS-OCR engine')
        plt.plot(x_new, y_new, label='Polynomial fit')  # Line 1
        # Format Plot
        plt.title("OCR engine resilience to %s" % target, fontsize=24)
        x_axis = target
        if target is 'blur':
            x_axis = 'Gaussian blur'
        plt.xlabel(x_axis, fontsize=16)
        plt.ylabel("OCR engine read-score (%)", fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend()
        plt.show()

    return


def alpha_sort(file):
    """
        Rewrites csv sorting row according to the GT values, alphabetically.
    """
    with open(file, encoding="utf8", errors='ignore') as csvFile:
        reader = csv.reader(csvFile)
        headers = next(reader, None)

        sorted_list = sorted(reader, key=lambda row: row[0].lower(), reverse=False)
        # for index, column in enumerate(sorted_list):
        # print(column)

    with open(file, 'w', encoding="utf8", errors='ignore') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(headers)
        writer.writerows(sorted_list)

    return


#file = os.path.join(os.getcwd(), 'PyCsvExp.csv')
#print(file)
#alpha_sort(file)
#plot_statistics(file, 'blur')

# print(similar("asdf", "asdg"))
# input("Press Enter to continue...")
