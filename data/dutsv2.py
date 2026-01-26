from .transforms import *
import os
import random
from glob import glob
from PIL import Image
import torchvision as tv
import torchvision.transforms.functional as TF


class TrainDUTSv2(torch.utils.data.Dataset):
    def __init__(self, root, clip_n):
        self.root = root
        img_dir = os.path.join(root, 'JPEGImages')
        flow_dir = os.path.join(root, 'JPEGFlows')
        mask_dir = os.path.join(root, 'Annotations')
        self.img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        self.flow_list = sorted(glob(os.path.join(flow_dir, '*.jpg')))
        self.mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.clip_n = clip_n
        self.to_tensor = tv.transforms.ToTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        all_frames = list(range(len(self.img_list)))
        frame_id = random.choice(all_frames)
        img_path = self.img_list[frame_id]
        flow_path = self.flow_list[frame_id]
        mask_path = self.mask_list[frame_id]
        img = Image.open(img_path).convert('RGB')
        flow = Image.open(flow_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # resize to 64p
        img = img.resize((64, 64), Image.BICUBIC)
        flow = flow.resize((64, 64), Image.BICUBIC)
        mask = mask.resize((64, 64), Image.BICUBIC)

        # joint flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            flow = TF.hflip(flow)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            flow = TF.vflip(flow)
            mask = TF.vflip(mask)

        # Convert formats.
        imgs = self.to_tensor(img).unsqueeze(0)   # [L=1, 3, H, W]
        flows = self.to_tensor(flow).unsqueeze(0)  # [L=1, 3, H, W]

        # Binary foreground mask as a long tensor: [L=1, H, W].
        masks = (self.to_tensor(mask)[0] > 0.5).long().unsqueeze(0)

        # Treat each DUTS image as its own "video" to avoid accidental positives in contrastive loss.
        video_id = os.path.splitext(os.path.basename(img_path))[0]
        return {"imgs": imgs, "flows": flows, "masks": masks, "video": video_id, "path": os.path.basename(img_path)}
