from torch import Tensor
from torchvision import models, transforms
import torch
import csv
from PIL import Image

f_photos = open("photos.txt","r")
photos = f_photos.readlines()


def img2tensor(pixel_img):
    normalize = transforms.Normalize(
        mean=[0.5657177752729754, 0.5381838567195789, 0.4972228365504561],
        std=[0.29023818639817184, 0.2874722565279285, 0.2933830104791508]
    )
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return preprocess(pixel_img)


def triplet_sampling(batch_size, csv_index):
    f_sim = open("triplet.csv","r")
    reader_csv = csv.reader(f_sim)

    reader = []
    count = 0
    for i in reader_csv:
        if count >= csv_index and count < csv_index + batch_size:
            reader.append(i)
        count += 1
    f_sim.close()

    ref_tensor = pos_tensor = neg_tensor = torch.Tensor(batch_size,3,384,256)

    sim_pos = []
    sim_neg = []

    count = 0
    for row in reader:
        ref_tensor[count] = img2tensor(Image.open(photos[int(row[0])].strip()))
        pos_tensor[count] = img2tensor(Image.open(photos[int(row[1])].strip()))
        neg_tensor[count] = img2tensor(Image.open(photos[int(row[2])].strip()))
        sim_pos.append(float(row[3]))
        sim_neg.append(float(row[4]))
        count += 1
        if count == batch_size:
            break

    img = [ref_tensor,pos_tensor,neg_tensor]
    sim = [sim_pos,sim_neg]
    #print("csv_index:\t"+str(csv_index))

    return img, sim, csv_index+batch_size
