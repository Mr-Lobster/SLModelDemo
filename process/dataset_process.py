import csv
import shutil
import os

currs = ['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34']


def dataLoader(csvFileName, imageFilePath, curr):
    csvFileName += curr
    csvFileName += '.csv'
    imageFilePath += curr
    newFileDIR = '/home/jian/swarm-learning/share/dataset/messidor_all'
    with open(csvFileName, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:
            print(row)
            currImage = row[0]
            filePath = os.path.join(imageFilePath, currImage)
            label = row[2]
            newFilePath = os.path.join(newFileDIR, label, currImage)
            shutil.copy(filePath, newFilePath)


csvFileName = '/home/jian/swarm-learning/share/dataset/messidor/Annotation_Base'
imageFilePath = '/home/jian/swarm-learning/share/dataset/messidor/Base'
for curr in currs:
    dataLoader(csvFileName, imageFilePath, curr)
