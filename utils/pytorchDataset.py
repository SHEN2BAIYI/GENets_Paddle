import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class PytorchDataset(Dataset):
    def __init__(self, root, transform, state='val'):
        # Init function
        self.root = root
        self.state = state
        self.transform = transform
        if state == 'train':
            self.data_dir = os.path.join(self.root, 'train_list.txt')
        elif state == 'val':
            self.data_dir = os.path.join(self.root, 'val_list.txt')
        self.data_list = self._read_txt(self.data_dir)

    def _read_txt(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            data_list = []
            lines = f.readlines()
            for line in lines:
                data_list.append(line.strip())
        return data_list

    def __getitem__(self, index):
        # Get the data information from the list.
        img_name, img_label = self.data_list[index].split(' ')

        img_path = os.path.join(self.root, self.state, img_name)
        img = self.transform(Image.open(img_path).convert('RGB'))
        img_data = np.array(img) / 255

        img_label = int(img_label)

        return img_data, img_label

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_list = [transforms.Resize(196),
                      transforms.CenterCrop(196), transforms.ToTensor(), transforms_normalize]
    transformer = transforms.Compose(transform_list)
    img_dataset = PytorchDataset('../data/ILSVRC2012mini', transformer)
    img_dataLoader = DataLoader(img_dataset, batch_size=64, pin_memory=True, shuffle=True, num_workers=8)

    for i, (img_data, img_label) in enumerate(img_dataLoader):
        print(i)
        print(img_data.shape)
        print(img_label.shape)


