import os
import random
import shutil

filePath = '/data/swarm-learning/dataset/'
labels = ['0', '1', '2', '3']
for l in labels:
    images = os.listdir(os.path.join(filePath, 'messidor_all', l))
    num = round(len(images) * 0.2)
    test = random.sample(images, num)
    train = list(set(images) - set(test))
    shutil.rmtree(os.path.join(filePath, 'test', l))
    os.makedirs(os.path.join(filePath, 'test', l))
    for item in test:
        shutil.copyfile(os.path.join(filePath, 'messidor_all', l, item), os.path.join(filePath, 'test', l, item))
    shutil.rmtree(os.path.join(filePath, 'train', l))
    os.makedirs(os.path.join(filePath, 'train', l))
    for item in train:
        shutil.copyfile(os.path.join(filePath, 'messidor_all', l, item), os.path.join(filePath, 'train', l, item))
