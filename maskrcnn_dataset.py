import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch

def upsize_mask(mask):
    size = 1    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[0, i, j] == 1:
                for x in range(max(0, i - size), min(mask.shape[0], i + size + 1)):
                    for y in range(max(0, j - size), min(mask.shape[1], j + size + 1)):
                        if np.sqrt((x - i)**2 + (y - j)**2) <= size:
                            mask[0, x, y] = 0.5
    mask[mask > 0] = 1
    return mask

def normalize_mask(mask):
    mask[mask > 0] = 1
    mask[mask <= 0] = 0
    return mask

class MaskRCNNDataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        if test:
            self.images = sorted([root_path+"/manual_test/"+i for i in os.listdir(root_path+"/manual_test/")])
            self.masks = sorted([root_path+"/manual_test_masks/"+i for i in os.listdir(root_path+"/manual_test_masks/")])
            for x in os.listdir(root_path+"/manual_test_masks/"):
                with open(root_path+"/manual_test_boxes/"+".".join(x.split('.')[:-1])+".txt", 'w') as f:
                    img = np.asarray(Image.open(root_path+"/manual_test_masks/"+x).convert("L"))
                    minx, maxx = img.shape[1] - 1, 0
                    miny, maxy = img.shape[0] - 1, 0
                    for i in range(img.shape[1]):
                        if np.sum(img[:, i]):
                            minx = i
                            break
                    for i in range(img.shape[1] - 1, -1, -1):
                        if np.sum(img[:, i]):
                            maxx = i
                            break
                    for i in range(img.shape[0]):
                        if np.sum(img[i, :]):
                            miny = i
                            break
                    for i in range(img.shape[0] - 1, -1, -1):
                        if np.sum(img[i, :]):
                            maxy = i
                            break
                    minx, maxx = min(minx, maxx), max(minx, maxx)
                    miny, maxy = min(miny, maxy), max(miny, maxy)
                    if minx == maxx:
                        print('ALARM!!!')
                        print(maxx)
                        print(root_path+"/manual_test_masks/"+x)
                    if miny == maxy:
                        print('ALARM!!!')
                        print(miny)
                        print(root_path+"/manual_test_masks/"+x)
                    f.write(f"{minx} {maxx} {miny} {maxy} {img.shape[1]} {img.shape[0]}")
            self.boxes = sorted([root_path+"/manual_test_boxes/"+i for i in os.listdir(root_path+"/manual_test_boxes/")])
            self.transform_mask = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()])
        else:
            self.images = sorted([root_path+"/train/"+i for i in os.listdir(root_path+"/train/")])
            self.masks = sorted([root_path+"/train_masks/"+i for i in os.listdir(root_path+"/train_masks/")])
            for x in os.listdir(root_path+"/train_masks/"):
                with open(root_path+"/train_boxes/"+".".join(x.split('.')[:-1])+".txt", 'w') as f:
                    img = np.asarray(Image.open(root_path+"/train_masks/"+x).convert("L"))
                    minx, maxx = img.shape[1] - 1, 0
                    miny, maxy = img.shape[0] - 1, 0
                    for i in range(img.shape[1]):
                        if np.sum(img[:, i]):
                            minx = i
                            break
                    for i in range(img.shape[1] - 1, -1, -1):
                        if np.sum(img[:, i]):
                            maxx = i
                            break
                    for i in range(img.shape[0]):
                        if np.sum(img[i, :]):
                            miny = i
                            break
                    for i in range(img.shape[0] - 1, -1, -1):
                        if np.sum(img[i, :]):
                            maxy = i
                            break
                    minx, maxx = min(minx, maxx), max(minx, maxx)
                    miny, maxy = min(miny, maxy), max(miny, maxy)
                    if minx == maxx:
                        print('ALARM!!!')
                        print(maxx)
                        print(root_path+"/train_masks/"+x)
                    if miny == maxy:
                        print('ALARM!!!')
                        print(miny)
                        print(root_path+"/train_masks/"+x)
                    f.write(f"{minx} {maxx} {miny} {maxy} {img.shape[1]} {img.shape[0]}")
            self.boxes = sorted([root_path+"/train_boxes/"+i for i in os.listdir(root_path+"/train_boxes/")])
            self.transform_mask = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                normalize_mask,
                upsize_mask])
        self.transform_img = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        with open(self.boxes[index], 'r') as f:
            boxes = list(map(int, f.read().split()))

        target = {}
        target["boxes"] = torch.tensor([[boxes[0] / boxes[4] * 256, boxes[2] / boxes[5] * 256, boxes[1] / boxes[4] * 256, boxes[3] / boxes[5] * 256]], dtype=torch.float32)
        target["labels"] = torch.tensor([1], dtype=torch.int64)
        target["masks"] = torch.as_tensor(self.transform_mask(mask), dtype=torch.uint8)
        target["image_id"] = torch.tensor([index])
        target["area"] = torch.tensor([(target["boxes"][0, 3] - target["boxes"][0, 1]) * (target["boxes"][0, 2] - target["boxes"][0, 0])])
        target["iscrowd"] = torch.zeros((1,), dtype=torch.int64)
        return self.transform_img(img), target

    def __len__(self):
        return len(self.images)
