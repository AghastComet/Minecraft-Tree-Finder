import torch
import torchvision
import transforms as T
from engine import train_one_epoch, evaluate
import utils
from torchdata.datapipes.iter import IterableWrapper, FileOpener
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import os
from PIL import Image

def get_name(path_and_stream):
    return os.path.basename(path_and_stream[0]), path_and_stream[1]

label_mappings = {
    'Acacia Tree': 1,
    'Birch Tree': 2,
    'Oak Tree': 3,
    'Spruce Tree': 4,
    'Jungle Tree': 5,
    'Dark Oak Tree': 6,
}

class TreeDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):

        datapipe1 = IterableWrapper(["labels.csv"])
        datapipe2 = FileOpener(datapipe1, mode='b')
        datapipe3 = datapipe2.map(get_name)
        csv_parser_dp = datapipe3.parse_csv(skip_lines=1)

        data = dict()
        for obj_type, x, y, dx, dy, filename, width, height in csv_parser_dp:
            x=int(x)
            y=int(y)
            dx=int(dx)
            dy=int(dy)
            if filename not in data:
                data[filename] = {'boxes': [], 'labels': [], 'area':[]}
            data[filename]['boxes'].append([x,y,x+dx,y+dy])
            data[filename]['labels'].append(label_mappings[obj_type])
            data[filename]['area'].append(dx*dy)

        for filename, filedata in data.items():
            data[filename]['boxes'] = torch.as_tensor(filedata['boxes'], dtype=torch.float32)
            data[filename]['labels'] = torch.as_tensor(filedata['labels'], dtype=torch.int64)
            data[filename]['area'] = torch.as_tensor(filedata['area'], dtype=torch.float32)
            data[filename]['iscrowd'] = torch.zeros((len(data),), dtype=torch.int64)
            data[filename]['image_id'] = torch.tensor([int(filename[:-4])])

        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = str(idx) + '.png'
        target = self.data[filename]
        img = Image.open(os.path.join('Images', filename)).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target




#backbone = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT).features
#backbone.out_channels = 1280
#anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

#roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

#model = FasterRCNN(backbone, num_classes=len(label_mappings)+1, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

def get_transforms(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

model = torch.load('model.pt')

model.eval()
dataset = TreeDataset(get_transforms(False))
r = model([dataset[0][0]])
#model([i[0] for i in TreeDataset()[:58]])
