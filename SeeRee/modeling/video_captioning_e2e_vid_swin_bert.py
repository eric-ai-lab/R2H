import torch
from fairscale.nn.misc import checkpoint_wrapper
import random


class VideoTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super(VideoTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length
        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        
        self.no_cos_mask = getattr(args, 'no_cos_mask', False)
        if self.learn_mask_enabled==True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length*args.max_img_seq_length,1) # Yue: attention
            self.sigmoid = torch.nn.Sigmoid()

            self.img_depended_att_full = torch.nn.Sequential(
            torch.nn.Linear(512, 32),
            # torch.nn.BatchNorm1d(32, eps=1e-12),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, args.max_img_seq_length),
            )
        
            self.img_depended_att_full_new = torch.nn.Sequential(
            torch.nn.Linear(1024, 32),
            # torch.nn.BatchNorm1d(32, eps=1e-12),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),
            )
            self.avgp = torch.nn.AvgPool2d(3)

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)
        #################################
        ############ here ###############
        #################################
        vid_feats_swin = self.swin(images)
        if self.use_grid_feat==True:
            vid_feats_swin = vid_feats_swin.permute(0, 2, 3, 4, 1)
        vid_feats_swin = vid_feats_swin.view(B, -1, self.latent_feat_size)
        vid_feats = self.fc(vid_feats_swin)

        


        # prepare VL transformer inputs
        kwargs['img_feats'] = vid_feats
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)
        # learn soft attention mask
        if self.learn_mask_enabled:
            

            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            vid_att_len = self.max_img_seq_length
            learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
            
            # swinbert paper
            # learn_att = self.sigmoid(learn_att)

            # Yue:
            # att_depended = self.img_depended_att_full(vid_feats)
            # att_depended = torch.mean(att_depended, axis=0) # 784*784
            # learn_att = self.sigmoid(learn_att)
            # learn_att = self.sigmoid(learn_att + att_depended)

            # yue: new:
            # att_depended = self.img_depended_att_full_new(vid_feats_swin) # B*784*1
            att_depended = self.img_depended_att_full(vid_feats)
            att_depended = att_depended.view(-1,1,784,784)
            learn_att = self.sigmoid(att_depended.squeeze())
            # learn_att = self.sigmoid(learn_att.unsqueeze(0) + att_depended.squeeze())
            

            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask)*learn_att
            learn_att = diag_mask + video_attention
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att>=0.5)*1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            if self.no_cos_mask:
                learn_att = torch.ones_like(learn_att).cuda()
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att

        #################################
        ############ here ###############
        #################################
        outputs = self.trans_encoder(*args, **kwargs)
        
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)  
            outputs = outputs + (loss_sparsity, )          
        return outputs
    
    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def diag_based_init_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        zeros_mask = torch.zeros_like(pretrained_learn_att)
        scale_factor = self.max_img_seq_length/pretrained_num_tokens
        
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 


    def bilinear_init_attn_mask(self, pretrain_attn_mask):
        print('init attn mask with bilinear interpolation')
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        scale_factor = int(self.max_img_seq_length/pretrained_num_tokens)
        sampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        with torch.no_grad():
            learn_att = sampler(pretrained_learn_att[None,None,:,:].double())[0,0,:,:].half()

    def random_init_attn_mask(self):
        print('random init attn mask')
        self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length*self.max_img_seq_length,1)


    def reload_attn_mask(self, pretrain_attn_mask): 
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze

 