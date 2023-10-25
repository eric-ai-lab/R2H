"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
from operator import is_
import os.path as op
import torch
from src.utils.comm import get_world_size
from .vision_language_tsv import (VisionLanguageTSVYamlDataset)
from .caption_tensorizer import build_tensorizer
from .data_sampler import DistributedSamplerLimited, NodeSplitSampler
from .data_utils.video_transforms import Compose, Resize, RandomCrop, ColorJitter, Normalize, CenterCrop, RandomHorizontalFlip, RandomResizedCrop
from .data_utils.volume_transforms import ClipToTensor

from src.utils.logger import LOGGER as logger
import cv2
import numpy as np
import json
import os
import re
from PIL import Image
def build_dataset(args, data_root, tokenizer, is_train=True, val_seen = False):
    logger.info(f'data_root:{data_root}')

    #Yue todo: later change yaml to something usable
    if not op.isfile(data_root):
        data_root = op.join(args.data_dir, data_root)
        # assert op.isfile(data_root), f"{data_root} does not exists"


    tensorizer = build_tensorizer(args, tokenizer, is_train=is_train)
    # dataset_class = VisionLanguageTSVYamlDataset
    # if not is_train and 'VATEX' in data_root:
    #     return dataset_class(args, data_root, tokenizer, tensorizer, is_train, args.on_memory)
    if 'CVDN' in data_root:
        if is_train == True:
            return Dataset_CVDN(args, data_root, tensorizer, 'train')
        elif is_train == False and val_seen == False:
            return Dataset_CVDN(args, data_root, tensorizer, 'val_unseen')
        else:
            return Dataset_CVDN(args, data_root, tensorizer, 'val_seen')
    elif 'Dialfred' in data_root:
        if is_train == True:
            return Dataset_Dialfred(args, data_root, tensorizer, 'train')
        elif is_train == False and val_seen == False:
            return Dataset_Dialfred(args, data_root, tensorizer, 'val_unseen')
        else:
            return Dataset_Dialfred(args, data_root, tensorizer, 'val_seen')
    elif 'AVDN' in data_root:
        if is_train == True:
            return Dataset_AVDN(args, data_root, tensorizer, 'train')
        elif is_train == False and val_seen == False:
            return Dataset_AVDN(args, data_root, tensorizer, 'val_unseen')
        else:
            return Dataset_AVDN(args, data_root, tensorizer, 'val_seen')

class Dataset_AVDN(torch.utils.data.Dataset):
    def __init__(self, args, data_root, tensorizer,is_train):
        super().__init__()
        
        self.split = is_train

        self.data_root = data_root
        # self.img = pickle.load(open('./_data/img_%s.pkl'%(self.args['dataset']), 'rb'))
        # self.txt = json.load(open('./_data/txt_%s.json'%(self.args['task']), 'r'))[self.split]
        self.root = data_root#'./commander_TATC/AVDN/'
        self.img_path = self.root + '%s/imgs'%self.split
        

        self.txt = json.load(open(self.root + '%s/%s.json'%(self.split,self.split), 'r'))


        # data_to_delete = []
        # for idx in range(len(self.txt)): 
        #     if (self.txt[idx]['len_images'] == 0):
        #         data_to_delete.append(self.txt[idx])
        #     if (not 'dialog_answer_parsed' in self.txt[idx].keys()):
        #         data_to_delete.append(self.txt[idx])

        # for i in data_to_delete:
        #     self.txt.remove(i)


        self.img_res = getattr(args, 'img_res', 224)
        # self.patch_size = getattr(args, 'patch_size', 16)
        self.decoder_num_frames = getattr(args, 'max_num_frames', 2) 

        self.raw_video_crop_list = [
                Resize(self.img_res),
                RandomCrop((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.43, 0.47 , 0.50],std=[0.22, 0.20, 0.19])
            ]
        self.raw_video_prcoess = Compose(self.raw_video_crop_list)
        self.tensorizer = tensorizer # includes tokenizer

        self.parsed = args.parsed
        self.parsed_with_seperation = args.parsed_with_seperation
        self.got_a_generate_b = args.got_a_generate_b
        self.qa_as_caption = args.qa_as_caption
        
        if self.parsed:
            self.parsing_result = json.load(open(self.root + 'AVDN_parsed.json', 'r'))
    def __len__(self):
        return len(self.txt)
    
    def __getitem__(self, idx):
        # item = self.txt[idx]
        
        # img = []
        # for b in self.img[item['video']]:
        #     img.append(self.str2img(b).unsqueeze(0))
        # img = torch.cat(img, dim=0)
        
        # txt, mask = self.str2txt(item['question'])
        
        # return img, txt, mask, item['answer']
        
        item = self.txt[idx]
        
        # img = []
        # for b in self.img[item['video']]:
        #     img.append(self.str2img(b).unsqueeze(0))
        # img = torch.cat(img, dim=0)
        image_sequence_key = item['map_name'] + '_'+ item['route_index']
        img = []
        
        for i in range(100): # set a max number
            try:
                img.append(
                    cv2.resize(cv2.imread(os.path.join(
                        self.img_path, image_sequence_key + '_' + str(i) +".jpg"), 1), (224,224))
                )
            except:
                break
        assert len(img) >0, self.img_path + image_sequence_key + '_' + str(i)
        for i in range(len(img), self.decoder_num_frames):
            img.append( np.zeros_like(img[0]) )
        img = torch.from_numpy(np.stack(img).transpose(0, 3, 1, 2))
        preproc_frames = self.apply_augmentations(img)
        
        if self.parsed:
            ans_txt = item['instructions'].split('[INS]')[-1]
            assert ans_txt in self.parsing_result.keys(), ans_txt
            ans_txt = self.parsing_result[ans_txt]
            
            
            if self.parsed_with_seperation:
                ans_txt = ans_txt.replace('1.', '')
                for i in range(2,10):
                    ans_txt = ans_txt.replace(str(i)+'.', 'and')
                # ans_txt = ans_txt.replace('stop')
            else:
                for i in range(10):
                    ans_txt = ans_txt.replace(str(i)+'.', '')
            ans_txt=re.sub(r'[^\w\s]', " ", ans_txt).lower().strip().replace('  ', ' ')
        else:
            ans_txt = item['instructions'].split('[INS]')[-1]
            ans_txt=re.sub('[^\w\s]'," ",ans_txt).lower().strip().replace('  ', ' ')
        if self.got_a_generate_b:
            question_txt = item['instructions'].split('[INS]')[0].split('[QUE]')[-1].lower()
            if question_txt.strip() == '':
                question_txt = 'what next'
            question_txt=re.sub(r'[^\w\s]', ' ',question_txt).strip().replace('  ', ' ')
            ###################################
            ######## Create Masks ##############
            example = self.tensorizer.tensorize_example_e2e(question_txt, preproc_frames, ans_txt, got_a_generate_b = self.got_a_generate_b, qa_as_caption = False, text_meta=None) # caption tensorizer
        
        else:
            example = self.tensorizer.tensorize_example_e2e(ans_txt, preproc_frames, got_a_generate_b = self.got_a_generate_b, qa_as_caption = self.qa_as_caption, text_meta=None) # caption tensorizer
        
        
        # tok, mask = self.str2txt(item['dialog_question']+ans_txt) 
        # # yue: finetune masking
        # idxs = [i for i in range(len(mask))]
        # np.random.shuffle(idxs)
        # mask_pos = idxs[:int(0.15*len(mask))]

        # masked_tok = []
        
        # for i in range(len(tok)):
        #     if i in mask_pos:
        #         masked_tok.append(tok[i])
        #         tok[i] = 103
        #     else:
        #         masked_tok.append(-1)
        
        return image_sequence_key, example, torch.tensor([1])
    
    def apply_augmentations(self, frames):
        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((self.decoder_num_frames,self.img_res,self.img_res,3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(self.decoder_num_frames):
            if num_of_frames==1: 
                # if it is from image-caption dataset, we duplicate the image
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[0]))
            else:
                # if it is from video-caption dataset, we add each frame to the list
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[i]))
        
        # adapt from torch_videovision: https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W) in the range [0, 1.0]
        crop_frames = self.raw_video_prcoess(frame_list)
        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames 
    
    def get_caption_file_in_coco_format(self):
        # for evaluation
        test_split = op.basename(self.data_root).split('.')[0]
        return op.join(self.root, self.split ,test_split + '_caption_coco_format.json')

class Dataset_CVDN(torch.utils.data.Dataset):
    def __init__(self, args, data_root, tensorizer,is_train):
        super().__init__()
        
        self.split = is_train

        self.data_root = data_root
        
        self.root = data_root#'./commander_TATC/CVDN/NDH_commander/'
        self.img_path = self.root + '%s/imgs'%self.split
        

        self.txt = json.load(open(self.root + '%s/%s.json'%(self.split,self.split), 'r'))


        data_to_delete = []
        for idx in range(len(self.txt)): 
            if (self.txt[idx]['len_images'] == 0):
                data_to_delete.append(self.txt[idx])
            if (not 'dialog_answer_parsed' in self.txt[idx].keys()):
                data_to_delete.append(self.txt[idx])
            # elif (self.txt[idx]['human_command_reliability'] == 0):
            #     data_to_delete.append(self.txt[idx])
        for i in data_to_delete:
            self.txt.remove(i)


        self.img_res = getattr(args, 'img_res', 224)
        # self.patch_size = getattr(args, 'patch_size', 16)
        self.decoder_num_frames = getattr(args, 'max_num_frames', 2) 

        self.raw_video_crop_list = [
                Resize(self.img_res),
                RandomCrop((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.43, 0.47 , 0.50],std=[0.22, 0.20, 0.19])
            ]
        self.raw_video_prcoess = Compose(self.raw_video_crop_list)
        self.tensorizer = tensorizer # includes tokenizer

        self.parsed = args.parsed
        self.parsed_with_seperation = args.parsed_with_seperation
        self.got_a_generate_b = args.got_a_generate_b
        self.qa_as_caption = args.qa_as_caption
    def __len__(self):
        return len(self.txt)
    
    def __getitem__(self, idx):
        # item = self.txt[idx]
        
        # img = []
        # for b in self.img[item['video']]:
        #     img.append(self.str2img(b).unsqueeze(0))
        # img = torch.cat(img, dim=0)
        
        # txt, mask = self.str2txt(item['question'])
        
        # return img, txt, mask, item['answer']
        
        item = self.txt[idx]
        
        # img = []
        # for b in self.img[item['video']]:
        #     img.append(self.str2img(b).unsqueeze(0))
        # img = torch.cat(img, dim=0)
        image_sequence_key = item['scan'] + '_'+ str(item['inst_idx'])
        img = []
        
        for i in range(item['len_images']):
            try:
                img.append(
                    cv2.resize(cv2.imread(os.path.join(self.img_path, item['scan'] \
                        + '_' + item['start_pano'] + '_'\
                        + str(item['inst_idx']) + '_' + '%02d'%i +".jpg"), 1), (224,224))
                )
            except:
                print(os.path.join(self.img_path, item['scan'] \
                        + '_' + item['start_pano'] + '_'\
                        + str(item['inst_idx']) + '_' + '%02d'%i +".jpg"))
        for i in range(item['len_images'], self.decoder_num_frames):
            img.append( np.zeros_like(img[0]) )
        img = torch.from_numpy(np.stack(img).transpose(0, 3, 1, 2))
        preproc_frames = self.apply_augmentations(img)
        
        if self.parsed:
            if 'dialog_answer_parsed' in item.keys():
                ans_txt = item['dialog_answer_parsed'].lower()
            else:
                ans_txt = item['dialog_answer'].lower() # R2R didn't get parsed
            
            if self.parsed_with_seperation:
                ans_txt = ans_txt.replace('1.', '')
                for i in range(2,10):
                    ans_txt = ans_txt.replace(str(i)+'.', 'and')
                # ans_txt = ans_txt.replace('stop')
            else:
                for i in range(10):
                    ans_txt = ans_txt.replace(str(i)+'.', '')
            ans_txt=re.sub('[^\w\s]'," ",ans_txt)
        else:
            ans_txt = item['dialog_answer'].lower()
            ans_txt=re.sub('[^\w\s]'," ",ans_txt)
        if self.got_a_generate_b:
            question_txt = item['dialog_question'].lower()
            # question_txt = 'what next'
            question_txt=re.sub('[^\w\s]'," ",question_txt)
            ###################################
            ######## Create Masks ##############
            example = self.tensorizer.tensorize_example_e2e(question_txt, preproc_frames, ans_txt, got_a_generate_b = self.got_a_generate_b, qa_as_caption = False, text_meta=None) # caption tensorizer
        
        else:
            example = self.tensorizer.tensorize_example_e2e(ans_txt, preproc_frames, got_a_generate_b = self.got_a_generate_b, qa_as_caption = self.qa_as_caption, text_meta=None) # caption tensorizer
        
        
        # tok, mask = self.str2txt(item['dialog_question']+ans_txt) 
        # # yue: finetune masking
        # idxs = [i for i in range(len(mask))]
        # np.random.shuffle(idxs)
        # mask_pos = idxs[:int(0.15*len(mask))]

        # masked_tok = []
        
        # for i in range(len(tok)):
        #     if i in mask_pos:
        #         masked_tok.append(tok[i])
        #         tok[i] = 103
        #     else:
        #         masked_tok.append(-1)
        
        return image_sequence_key, example, torch.tensor([1])
    
    def apply_augmentations(self, frames):
        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((self.decoder_num_frames,self.img_res,self.img_res,3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(self.decoder_num_frames):
            if num_of_frames==1: 
                # if it is from image-caption dataset, we duplicate the image
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[0]))
            else:
                # if it is from video-caption dataset, we add each frame to the list
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[i]))
        
        # adapt from torch_videovision: https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W) in the range [0, 1.0]
        crop_frames = self.raw_video_prcoess(frame_list)
        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames 
    
    def get_caption_file_in_coco_format(self):
        # for evaluation
        test_split = op.basename(self.data_root).split('.')[0]
        return op.join(self.root, self.split ,test_split + '_caption_coco_format.json')

class Dataset_Dialfred(torch.utils.data.Dataset):
    def __init__(self, args, data_root, tensorizer,is_train):
        super().__init__()
        
        self.split = is_train

        self.data_root = data_root
        # self.img = pickle.load(open('./_data/img_%s.pkl'%(self.args['dataset']), 'rb'))
        # self.txt = json.load(open('./_data/txt_%s.json'%(self.args['task']), 'r'))[self.split]
        self.root = data_root #'./commander_TATC/Dialfred/'
        self.img_path = self.root + '%s/'%self.split
        

        self.txt = json.load(open(self.root + '%s/%s.json'%(self.split,self.split), 'r'))


        data_to_delete = []
        for idx in range(len(self.txt)): 
            if (self.txt[idx]['dialog_answer'] == None):
                data_to_delete.append(self.txt[idx])
            # elif (self.txt[idx]['human_command_reliability'] == 0):
            #     data_to_delete.append(self.txt[idx])
        for i in data_to_delete:
            self.txt.remove(i)


        self.img_res = getattr(args, 'img_res', 224)
        # self.patch_size = getattr(args, 'patch_size', 16)
        self.decoder_num_frames = getattr(args, 'max_num_frames', 2) 

        self.raw_video_crop_list = [
                Resize(self.img_res),
                RandomCrop((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.43, 0.47 , 0.50],std=[0.22, 0.20, 0.19])
            ]
        self.raw_video_prcoess = Compose(self.raw_video_crop_list)
        self.tensorizer = tensorizer # includes tokenizer

        self.parsed = args.parsed
        self.parsed_with_seperation = args.parsed_with_seperation
        self.got_a_generate_b = args.got_a_generate_b
        self.qa_as_caption = args.qa_as_caption
    def __len__(self):
        return len(self.txt)
    
    def __getitem__(self, idx):

        
        item = self.txt[idx]
        image_sequence_key = item['task'] + '/'+ item['trial'] 
        img = []
        
        for img_name in item['images']:
            try:
                img.append(
                    cv2.resize(cv2.imread(
                        os.path.join(self.img_path, image_sequence_key, 'raw_images', img_name), 1
                        ), (224,224))
                )
            except:
                print(os.path.join(self.img_path, image_sequence_key, 'raw_images', img_name))
        for i in range(len(item['images']), self.decoder_num_frames):
            img.append( np.zeros_like(img[0]) )
        img = torch.from_numpy(np.stack(img).transpose(0, 3, 1, 2))
        preproc_frames = self.apply_augmentations(img)

        try:
            ans_txt = item['dialog_answer'].lower()
        except:
            print(item['dialog_answer'])
        ans_txt=re.sub('[^\w\s]'," ",ans_txt)

        if self.got_a_generate_b:
            question_txt = item['dialog_question'].lower().replace('<<', '').replace(">>", ' ')
            # question_txt = 'what next'
            question_txt=re.sub('[^\w\s]'," ",question_txt)
            ###################################
            ######## Create Masks ##############
            example = self.tensorizer.tensorize_example_e2e(question_txt, preproc_frames, ans_txt, got_a_generate_b = self.got_a_generate_b, qa_as_caption = False, text_meta=None) # caption tensorizer
        
        else:
            example = self.tensorizer.tensorize_example_e2e(ans_txt, preproc_frames, got_a_generate_b = self.got_a_generate_b, qa_as_caption = self.qa_as_caption, text_meta=None) # caption tensorizer
        
        return image_sequence_key + '_' + str(item['idx']), example, torch.tensor([1])
    
    def apply_augmentations(self, frames):
        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((self.decoder_num_frames,self.img_res,self.img_res,3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(self.decoder_num_frames):
            if num_of_frames==1: 
                # if it is from image-caption dataset, we duplicate the image
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[0]))
            else:
                # if it is from video-caption dataset, we add each frame to the list
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[i]))
        
        # adapt from torch_videovision: https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W) in the range [0, 1.0]
        crop_frames = self.raw_video_prcoess(frame_list)
        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames 
    
    def get_caption_file_in_coco_format(self):
        # for evaluation
        test_split = op.basename(self.data_root).split('.')[0]
        return op.join(self.root, self.split ,test_split + '_caption_coco_format.json')


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_batch_data_sampler(sampler, images_per_gpu, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_sampler(dataset, shuffle, distributed, random_seed, limited_samples=-1):
    if distributed:
        if dataset.is_composite:
            # first_epoch_skip_shuffle not working yet
            logger.info("Enable NodeSplitSampler with first_epoch_skip_shuffle=True")
            return NodeSplitSampler(
                dataset, shuffle=shuffle, random_seed=random_seed,
                first_epoch_skip_shuffle=True)
        elif limited_samples < 1:
            return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, seed=random_seed)
        else:  # use limited distributed sampler
            return DistributedSamplerLimited(dataset, shuffle=shuffle, limited=limited_samples)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args, data_root, tokenizer, is_distributed=True,
        is_train=True, start_iter=0, num_gpus=8, val_seen=False):

    dataset = build_dataset(args, data_root, tokenizer, is_train=is_train, val_seen=val_seen)
    if is_train==True:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_epoch = len(dataset) // images_per_batch
        num_iters = iters_per_epoch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    if hasattr(args, 'limited_samples'):
        limited_samples = args.limited_samples // num_gpus
    else:
        limited_samples = -1
    random_seed = args.seed
    sampler = make_data_sampler(
        dataset, shuffle, is_distributed, limited_samples=limited_samples,
        random_seed=random_seed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True, worker_init_fn=init_seeds,
    )

    

    return data_loader

def init_seeds(seed=88):
    import os, random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)