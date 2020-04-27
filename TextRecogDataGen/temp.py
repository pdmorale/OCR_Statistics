import os, csv
from matplotlib import pyplot as plt
import numpy as np
import random as rnd


def plot_statistics(file, target):
    with open(file, encoding="utf-8_sig", errors='ignore') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = np.array(list(reader))

    print(headers)
    print(data.shape)
    print(data[:3])

    # Plot the data
    plt.plot(float(data[:, 18]), float(data[:, 35]))
    plt.axis('equal')
    plt.xlabel(headers[18])
    plt.ylabel(headers[35])
    plt.show()
    return


def gen_plot(file, target):
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
            effect = float(row[headers.index(target)])  # Convert to int
            score = float(row[last])
            # print("scored: ", score, "  | blur: ", effect)
            feature.append(effect)  # appending feature
            measure.append(score)

    # calculate polynomials
    z = np.polyfit(feature, measure, 18)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(feature[0], feature[-1], 50)
    y_new = f(x_new)

    return feature, measure, x_new, y_new


def plot_statistics2(file, file2, target):
    feature, measure, x_new, y_new = gen_plot(file, target)
    feature1, measure1, x_new1, y_new1 = gen_plot(file2, target)

    # Plot Data
    fig = plt.figure(dpi=128, figsize=(10, 6))
    plt.scatter(feature, measure, c='green', label='MS-OCR legacy')
    plt.scatter(feature1, measure1, c='red', label='MS-OCR none')
    plt.plot(x_new, y_new, label='Poly-fit legacy')  # Line 1
    plt.plot(x_new1, y_new1, label='Poly-fit none')  # Line 1

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


def average_Yvalues(x_array, y_array):
    unique_feature, index, count = np.unique(x_array, return_index=True, return_counts=True)
    duplicates = unique_feature[count > 1]

    u_index = list(index[:])
    d_index = []
    for _ in sorted(index):
        if feature[_] in duplicates:
            u_index.remove(_)
            d_index.append(_)

    triplet = []
    for _ in range(len(x_array)):  #
        if feature[_] in duplicates:
            triplet.append([_, x_array[_], y_array[_]])
            print(_, " | ", x_array[_], " | ", y_array[_])

    av, counter = 0, 0
    feature_av, measure_av = [], []
    for i in range(len(triplet)):
        # a.append(triplet[i][2])
        av += triplet[i][2]
        counter += 1
        if i is len(triplet) - 1:
            av /= counter
            feature_av.append(triplet[i][1])
            measure_av.append(av)
        elif triplet[i][1] != triplet[i + 1][1]:
            av /= counter
            feature_av.append(triplet[i][1])
            measure_av.append(av)
            av = 0
            counter = 0

    return feature_av, measure_av, d_index, index


# file = os.path.join(os.getcwd(), 'TextRecogDataGen/PyCsvExp.csv')
file2 = os.path.join(os.getcwd(), 'TextRecogDataGen/PyCsvExp3.csv')
# plot_statistics2(file, file2, 'blur')

fileNLP = os.path.join(os.getcwd(), 'TextRecogDataGen/PyCsvExpRandom.csv')
fileNLPr = os.path.join(os.getcwd(), 'TextRecogDataGen/PyCsvExpRandom4.csv')
feature, measure, x_new, y_new = gen_plot(fileNLP, 'length')
featurer, measurer, xr_new, yr_new = gen_plot(fileNLPr, 'length')

feature_d, measure_av, d_index, index = average_Yvalues(feature, measure)
feature_dr, measure_avr, d_indexr, indexr = average_Yvalues(featurer, measurer)



print(d_index)
print(list(index))


# for i in range(len(index)):

cc = 0
measure_f = []
feature_f = []
for _ in index:  # index
    feature_f.append(feature[_])
    if _ in d_index:
        measure_f.append(measure_av[cc])
        cc += 1
    else:
        measure_f.append(measure[_])


cc = 0
measure_fr = []
feature_fr = []
for _ in indexr:  # index
    feature_fr.append(featurer[_])
    if _ in d_indexr:
        measure_fr.append(measure_avr[cc])
        cc += 1
    else:
        measure_fr.append(measurer[_])


print(len(measure_f), len(feature_f))
print(feature_f)

ratio = []
for _ in range(len(feature_f)):
    ratio.append(measure_fr[_]/measure_f[_])

zf = np.polyfit(feature_f, ratio, 18)
ff = np.poly1d(zf)

# calculate new x's and y's
x_newf = np.linspace(feature_f[0], feature_f[-1], 50)
y_newf = ff(x_new)

# Plot Data
fig = plt.figure(dpi=128, figsize=(10, 6))
plt.scatter(feature_f, ratio, c='green', label='Omnipage-OCR None')
# plt.scatter(featurer, measurer, c='blue', marker='^', label='Omnipage-OCR None random')
plt.plot(x_newf, y_newf, label='Polynomial fit')  # Line 1

# Format Plot
plt.title("Omnipage-OCR, performace ratio (nr/r)", fontsize=24)
x_axis = 'Sentence length [number of chars]'
plt.xlabel(x_axis, fontsize=16)
plt.ylabel("Ratio [nr/r]", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.ylim(ymin=0)
plt.legend()
#plt.show()

fileDic = os.path.join(os.getcwd(), 'TextRecogDataGen/dicts/google_en.txt')
with open(fileDic, 'r', encoding="utf8", errors='ignore') as d:
    lang_dict = [l for l in d.read().splitlines() if len(l) > 0]

#shuffle_dict = lang_dict[:]

#print(shuffle_dict[13])

sentence = ''
WordVEC = [1,3,3,6,4]

for n in WordVEC:
#    rnd.shuffle(shuffle_dict)
    rndword = rnd.choice(lang_dict)
    while len(rndword) != n:
        rndword = rnd.choice(lang_dict)
    sentence += rndword + ' '


print(sentence[:-1])

plt.show()

def exp(file):
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    blur = data[:, 19]
    score = data[:, 36]
    print(blur)
    return
