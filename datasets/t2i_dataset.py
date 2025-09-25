import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import linecache
from collections import defaultdict
import json
from torchvision import transforms
from models import VLChatProcessor
from torch.utils.data import Dataset, DataLoader

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class TextToImageDataset(Dataset):
    def __init__(
        self,
        model_path,
    ):
        self.gen_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.selfcheck_text = '<end_of_image>\nLet me think Does this image match the prompt...'
        self.data = []
        self.internvl_data = []
        self.num_gen_image_tokens = 576
        self.max_reason_length = 184
        self.max_prompt_length = 200
        self.internvl_len = 0
        self.load_from_internvl()

    def load_from_internvl(self):
        self.internvl_data = []
        self.thre = 0.7
        paths = [
            "data/t2i_examples/"
        ]
        imageid = 'flux'
        for root_path in paths:
            label_path = os.path.join(root_path, 'labels', f'{imageid}_{self.thre}.jsonl')
            num = len(linecache.getlines(label_path))
            for i in range(num):
                curcontent = json.loads(linecache.getline(label_path, i+1))
                prompt = curcontent['prompt']
                curdatalist = curcontent['data']
                for curdata in curdatalist:
                    self.internvl_data.append((prompt, curdata['img_path']))
        self.internvl_len = len(self.internvl_data)
        self.data.extend(self.internvl_data)
        
    def __getitem__(self, idx):
        curdata = self.data[idx]
        image = Image.open(curdata[1]).convert('RGB')
        image = self.gen_transform(image)
        conversation = [
            {
                "role": "<|User|>",
                "content":curdata[0],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', max_length=self.max_prompt_length, truncation=True).squeeze(0)
        if random.random() < 0.1:
            input_ids[1:-1] = self.vl_chat_processor.pad_id

        return {"input_ids": input_ids, "image": image}

    def __len__(self):
        return self.internvl_len

def my_collate_fn(batch):
    return batch

def TextToImageDataloader(cfg):
    dataset = TextToImageDataset(
        model_path=cfg.model.processor_path,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.train.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        drop_last=True,
        collate_fn=my_collate_fn
    )
    
    return loader
