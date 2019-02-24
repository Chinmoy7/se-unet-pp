import numpy as np
import os, glob, datetime, time, sys, random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import scipy.io as sio
import h5py

class CarvanaDataset(Dataset):
    def __init__(self, root, resize=False, resize_h=None, resize_w=None, crop=False, crop_h=None, crop_w=None,
                     hflip=False, vflip=False):
        super(Dataset, self).__init__()
        self.root = root+'images/'
        self.img_paths = os.listdir(self.root)
        self.resize = resize
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.crop = crop
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.hflip = hflip
        self.vflip = vflip


    def __len__(self):
        return len(self.img_paths)

    # Use this transform function to apply identical transformations on both image and label

    def transform(self, image, mask, resize=False, resize_h=None, resize_w=None, crop=False, crop_h=None, crop_w=None,
                     hflip=False, vflip=False):

        # Resize
        if(resize==True):
            resize = transforms.Resize(size=(resize_h, resize_w))
            image = resize(image)
            mask = resize(mask)

        # Random crop
        if(crop==True):
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(crop_h, crop_w))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if(hflip==True):
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        # Random vertical flipping
        if(vflip==True):
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        # mask = TF.to_tensor(mask)

        return image, mask

    def __getitem__(self, index):

        # Prepare image and label paths
        img_path = self.root+self.img_paths[index]
        lbl_path_temp = img_path[:-4]+'_mask.gif'
        lbl_path = lbl_path_temp.replace("images", "labels")
        
        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        
        # Preprocessing
        img,lbl= self.transform(img,lbl,resize=self.resize, resize_h=self.resize_h, resize_w=self.resize_w, crop=self.crop, crop_h=self.crop_h, crop_w=self.crop_w,
                                hflip=self.hflip, vflip=self.vflip)

        # Return image and label
        return img, torch.from_numpy(np.asarray(lbl)).unsqueeze(0).float()

class LiTSDataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        super(Dataset, self).__init__()
        self.root = root
        self.train = train
        self.paths = []
        for i in range(1, len(os.listdir(self.root))+1):
            temp_dir = os.listdir(self.root+str(i)+'/')
            for j in range(len(temp_dir)): 
                temp_dir[j] = self.root+str(i)+'/'+temp_dir[j]
            self.paths.append(temp_dir)
        self.img_paths = [item for sublist in self.paths for item in sublist]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

# Use this transform function to apply identical transformations on both image and label

    # def transform(self, image, mask):

    #     # Resize
    #     # resize = transforms.Resize(size=(256, 256))
    #     # image = resize(image)
    #     # mask = resize(mask)

    #     # Random crop
    #     i, j, h, w = transforms.RandomCrop.get_params(
    #         image, output_size=(512, 512))
    #     image = TF.crop(image, i, j, h, w)
    #     mask = TF.crop(mask, i, j, h, w)

    #     # Random horizontal flipping
    #     if random.random() > 0.5:
    #         image = TF.hflip(image)
    #         mask = TF.hflip(mask)

    #     # Random vertical flipping
    #     if random.random() > 0.5:
    #         image = TF.vflip(image)
    #         mask = TF.vflip(mask)

    #     # Transform to tensor
    #     # image = TF.to_tensor(image)
    #     # mask = TF.to_tensor(mask)
    #     return image, mask

    def __getitem__(self, index):

        # Prepare image and label paths
        img_path = self.img_paths[index]
        # if(self.train):
        lbl_path = img_path.replace("images_volumes", "liver_seg").replace("mat","png")
        # else:
            # lbl_path = img_path.replace("valid_hq", "valid_masks")

        # Read image and label
        # input_img = sitk.ReadImage(img_path)
        # input_lbl = sitk.ReadImage(lbl_path)
        # img_arr = sitk.GetArrayFromImage(input_img)
        # lbl_arr = sitk.GetArrayFromImage(input_lbl)
        # try:
        #     f = h5py.File(img_path,'r')
        # except(OSError):
        #     pass
        # img = f.get('data/variable1')
        # print(img_path)
        # sys.stdout.flush()
        f = sio.loadmat(img_path)
        img = f['section']
        lbl = Image.open(lbl_path)
        # img_arr = np.array(img)
        # lbl_arr = np.array(lbl)

        # img=cv2.imread(img_path)
        # lbl=cv2.imread(lbl_path, 0)

        # stack = np.dstack((img, lbl))

        # stack_img = Image.fromarray(stack)
        
        # Preprocessing
        if(self.transforms is not None):
            # stack_img = self.transforms(stack_img)
            img= self.transforms(img)
            lbl= self.transforms(lbl)

        img_arr = np.asarray(img)
        lbl_arr = np.asarray(lbl)
        # img_arr = np.hstack((np.zeros((img_arr.shape[0],1,img_arr.shape[2])), img_arr, np.zeros((img_arr.shape[0],1,img_arr.shape[2]))))
        # lbl_arr = np.hstack((np.zeros((lbl_arr.shape[0],1)), lbl_arr, np.zeros((lbl_arr.shape[0],1))))
        # stack_arr = np.array(stack_img)
            # img = self.transforms(img)
            # lbl = self.transforms(lbl)
        # top_fill = np.zeros((11, 512, 512), dtype=int)
        # bottom_fill = np.zeros((10,512,512), dtype=int)
        # # mod_img_arr = np.hstack((top_fill, img_arr, bottom_fill))
        # mod_img_arr = np.vstack((top_fill, img_arr, bottom_fill))
        # mod_lbl_arr = np.vstack((top_fill, lbl_arr, bottom_fill))
        # rshp_img = np.reshape(mod_img_arr, (-1,1,64,64,64))
        # rshp_lbl = np.reshape(mod_lbl_arr, (-1,1,64,64,64))
        # # proc_img = split_2d(mod_img_arr, 3, 8, 8)
        # # proc_lbl = split_2d(mod_lbl_arr, 3, 8, 8)

        # Return image and label
        return torch.from_numpy(img_arr/255.0).unsqueeze(0).float(), torch.from_numpy(lbl_arr/255.0).unsqueeze(0).float()
        # return torch.from_numpy(proc_img).float(), torch.from_numpy(proc_lbl).float()

def get_train_valid_loader(data_dir, batch_size, random_seed, valid_size=0.01, shuffle=True, num_workers=4,
                            pin_memory=False,train=True, resize=False, resize_h=None, resize_w=None, crop=False,
                            crop_h=None, crop_w=None, hflip=False, vflip=False, to_tensor=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # load the dataset
    # train_dataset = datasets.CIFAR10(
    #     root=data_dir, train=True,
    #     download=True, transform=train_transform,
    # )

    # valid_dataset = datasets.CIFAR10(
    #     root=data_dir, train=True,
    #     download=True, transform=valid_transform,
    # )

    train_dataset = CarvanaDataset(
                    root=data_dir, resize=resize, resize_h=resize_h, resize_w=resize_w, crop=crop,
                    crop_h=crop_h, crop_w=crop_w, hflip=hflip, vflip=vflip
                    )

    valid_dataset = CarvanaDataset(
                    root=data_dir, resize=resize, resize_h=resize_h, resize_w=resize_w, crop=crop,
                    crop_h=crop_h, crop_w=crop_w, hflip=hflip, vflip=vflip
                    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=pin_memory,
                    )
    valid_loader = DataLoader(
                    valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers, pin_memory=pin_memory,
                    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir, batch_size, shuffle=False, num_workers=4, pin_memory=False, train=False,
                    resize=False, resize_h=None, resize_w=None):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    # dataset = datasets.CIFAR10(
        # root=data_dir, train=False,
        # download=True, transform=transform,
    # )
    dataset = CarvanaDataset(data_dir, resize=resize, resize_h=resize_h, resize_w=resize_w)

    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=shuffle,
    #     num_workers=num_workers, pin_memory=pin_memory,
    # )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader
    # return dataset