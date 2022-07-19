import os
from torch.utils.data import Dataset
from PIL import Image


class CelebADataset(Dataset):
    def __init__(self, img_dir, train=True, ratio=0.8, transform=None): 
        self.img_dir = img_dir
        self.train = train
        self.ratio = ratio
        filelist = list([(self.img_dir+f) for f in sorted(os.listdir(self.img_dir))])
        ell = len(filelist)
        if train:
            self.file_list = filelist[:int(ell * self.ratio)]
        else: 
            self.file_list = filelist[int(ell * self.ratio):]
        self.data_length = len(self.file_list)
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        data = Image.open(self.file_list[item]).convert("RGB")
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def get_file_list(self):
        return self.file_list
    
    def get_data_length(self):
        return self.data_length