import os
import torch
import numpy as np
from PIL import Image
import PIL
PIL.PILLOW_VERSION = PIL.__version__  # workaround for ImportError: cannot import name 'PILLOW_VERSION' from 'PIL'
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
from class_grouping_traffic_signs import get_group, get_label_index
import cv2

class MNISTRGB(MNIST):
    """MNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNIST_Filter(MNISTRGB):
    """MNIST Dataset.
    """
    def __Filter__(self, known):
        targets = self.targets.data.numpy()
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)

class MNIST_OSR(object):
    def __init__(self, known, dataroot='./data/mnist', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = MNIST_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class CIFAR10_Filter(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR10_OSR(object):
    def __init__(self, known, dataroot='./data/cifar10', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class CIFAR100_Filter(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)

class CIFAR100_OSR(object):
    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))

        print('Selected Labels: ', known)

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False
        
        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )


class SVHN_Filter(SVHN):
    """SVHN Dataset.
    """
    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.labels = self.data[mask], np.array(new_targets)

class SVHN_OSR(object):
    def __init__(self, known, dataroot='./data/svhn', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = SVHN_Filter(root=dataroot, split='train', download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

class Tiny_ImageNet_OSR(object):
    def __init__(self, known, dataroot='./data/tiny_imagenet', use_gpu=True, num_workers=8, batch_size=128, img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 200))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


def prepare_data(dataroot, subdirectory, empty_classes, classes_with_subfolders, do_class_grouping):
    data = []
    folder_list = glob.glob(osp.join(dataroot, subdirectory, "*"))
    for class_path in folder_list:
        class_name = class_path.split("/")[-1]
        # some classes are grouped for training. Get group name in that case. Unless for unknown data
        if do_class_grouping:
            class_name = get_group(class_name)
        # empty classes
        if class_name in empty_classes:  # Skip folders that do not contain training images
            continue
        # classes with subfolders
        elif class_name in classes_with_subfolders:
            folder_list_subfolders = glob.glob(osp.join(dataroot, subdirectory, class_name, "*"))
            for subclass_path in folder_list_subfolders:
                # if subclass_path is a folder, get images from that folder
                if osp.isdir(subclass_path):
                    for img_path in glob.glob(subclass_path + "/*"):
                        if not img_path.endswith(('jpeg', 'jpg', 'png', 'ppm')):
                            continue
                        data.append([img_path, class_name])
                else:
                    # if subclass_path is an image, get the images
                    if not img_path.endswith(('jpeg', 'jpg', 'png', 'ppm')):
                        continue
                    data.append([img_path, class_name])
        # normal folders
        else:
            for img_path in glob.glob(class_path + "/*"):
                if not img_path.endswith(('jpeg', 'jpg', 'png', 'ppm')):
                    continue
                data.append([img_path, class_name])
    return data

# class VialyticsTrafficSigns_Filter(ImageFolder):
#     def __Filter__(self, dataroot, data_loader_type, empty_training_classes, training_classes_with_subfolders,
#                  empty_test_classes, test_classes_with_subfolders):
#         if data_loader_type == 'train':
#             data = prepare_data(dataroot=dataroot, subdirectory='train/known', empty_classes=empty_training_classes,
#                                 classes_with_subfolders=training_classes_with_subfolders)
#
#         elif data_loader_type == 'test':
#             data = prepare_data(dataroot=dataroot, subdirectory='test/known', empty_classes=[],
#                                 classes_with_subfolders=[])
#
#         # unknown data consists of background classes (not used in training) and unknown classes
#         elif data_loader_type == 'out':
#             data = prepare_data(dataroot=dataroot, subdirectory='test/unknown', empty_classes=empty_test_classes,
#                                 classes_with_subfolders=test_classes_with_subfolders)
#             data += prepare_data(dataroot=dataroot, subdirectory='test/background', empty_classes=[],
#                                  classes_with_subfolders=[])
#
#         img_paths, targets = [], []
#         label_index = get_label_index(include_background_training_classes=False)
#         for img_paths_and_target in data:
#             img_paths.append(img_paths_and_target[0])
#             class_id = label_index[img_paths_and_target[1]]
#             targets.append(torch.tensor([class_id]))
#
#         self.img_paths, self.targets = img_paths, targets


#To do: What about background classes?
class VialyticsTrafficSigns_OSR(Dataset):
    def __init__(self, dataroot, data_loader_type, empty_classes, classes_with_subfolders, img_size):

        if data_loader_type == 'train':
            self.data = prepare_data(dataroot=dataroot, subdirectory='train/known', empty_classes=empty_classes,
                                     classes_with_subfolders=classes_with_subfolders, do_class_grouping=True)

        elif data_loader_type == 'test':
            self.data = prepare_data(dataroot=dataroot, subdirectory='test/known', empty_classes=[],
                                     classes_with_subfolders=[], do_class_grouping=True)

        # unknown data consists of background classes (not used in training) and unknown classes
        elif data_loader_type == 'out':
            self.data = prepare_data(dataroot=dataroot, subdirectory='test/unknown', empty_classes=empty_classes,
                                     classes_with_subfolders=classes_with_subfolders, do_class_grouping=False)
            self.data += prepare_data(dataroot=dataroot, subdirectory='test/background', empty_classes=[],
                                      classes_with_subfolders=[], do_class_grouping=False)

        self.label_index = get_label_index(include_background_training_classes=True, include_unknown_classes=True)
        self.img_dim = (img_size, img_size)  # format: (w, h)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.label_index[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor(class_id)
        return img_tensor.float(), class_id

        # train_transform = transforms.Compose([
        #     transforms.Resize((img_size, img_size)),
        #     transforms.ToTensor(),
        # ])
        #
        # transform = transforms.Compose([
        #     transforms.Resize((img_size, img_size)),
        #     transforms.ToTensor(),
        # ])


