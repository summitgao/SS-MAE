""" 
Masked Autoencoders Are Scalable Vision Learners

"""


import torch
import torch.nn as nn
import torch.nn.functional as F 
from net.lidar_feature_extractor import lidar_e
from net.hsi_feature_extractor import hsi_e
from net.VIT.layers.patch_embd import PatchEmbed_spa,PatchEmbed_chan
from net.VIT.vit import VisionTransformer


class MaskTransLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct the normalization for each patchs
        """
        super(MaskTransLayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
       
    def forward(self, x):
        u = x[:, :].mean(-1, keepdim=True)
        s = (x[:, :] - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class MAEVisionTransformers(nn.Module):
    def __init__(self,
                 channel_number=145,
                 img_size = 224,
                 patch_size = 16,
                 encoder_dim = 512,
                 encoder_depth = 24,
                 encoder_heads = 16,
                 decoder_dim = 512, 
                 decoder_depth = 8, 
                 decoder_heads = 16, 
                 mask_ratio = 0.75,
                 args = None
                 ) :
        super().__init__()
        self.patch_size = patch_size
        self.num_patch = (img_size // self.patch_size, img_size // self.patch_size)

        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        base_cfg = dict(
            img_size=img_size, 
            in_chans=3,
            num_classes=15,
            mlp_ratio=4., 
            qkv_bias=True,
            drop_rate = 0.,
            attn_drop_rate = 0.,
            drop_path_rate = 0., 
            embed_layer_spa=PatchEmbed_spa,
            embed_layer_chan=PatchEmbed_chan,
            pos_embed="cosine", 
            norm_layer=nn.LayerNorm, 
            act_layer=nn.GELU, 
            pool='mean',
            args = args
        )
        encoder_model_dict = dict(
            in_chans = channel_number,
            patch_size = self.patch_size,
            embed_dim=encoder_dim, 
            depth=encoder_depth, 
            num_heads=encoder_heads,
            classification=False,
            vit_type="encoder",
            mask_ratio = mask_ratio,
            args = args
        )
        decoder_model_dict = dict(
            patch_size = self.patch_size,
            in_chans=channel_number,
            embed_dim=decoder_dim, 
            depth=decoder_depth, 
            num_heads=decoder_heads,
            classification=False,
            vit_type="decoder",
            mask_ratio = mask_ratio,
            args = args
        )
        
        ENCODER_MODEL_CFG = {**base_cfg, **encoder_model_dict}
        DECODER_MODEL_CFG = {**base_cfg, **decoder_model_dict}
        
        # vit embeeding 
        self.Encoder = VisionTransformer(**ENCODER_MODEL_CFG)
        self.Decoder = VisionTransformer(**DECODER_MODEL_CFG)
        self.Encoder.to(self.device)
        self.Decoder.to(self.device)
        output_dim_spa = channel_number
        output_dim_chan = img_size*img_size
        # project encoder embeeding to decoder embeeding
        self.proj_spa = nn.Linear(encoder_dim, decoder_dim)
        self.proj_chan = nn.Linear(encoder_dim, decoder_dim)
        self.restruction_spa = nn.Linear(decoder_dim, output_dim_spa)
        self.restruction_chan = nn.Linear(decoder_dim, output_dim_chan)
        self.norm_spa = nn.LayerNorm(output_dim_spa)
        self.norm_chan = nn.LayerNorm(output_dim_chan)
        self.patch_norm_spa = MaskTransLayerNorm(output_dim_spa)
        self.patch_norm_chan = MaskTransLayerNorm(output_dim_chan)
        # restore image from unconv
        self.unconv_spa = nn.ConvTranspose2d(output_dim_spa, channel_number, patch_size, patch_size)
        self.unconv_chan = nn.Linear(self.num_patch[0]*self.num_patch[0], self.num_patch[0]*self.num_patch[0])
        self.apply(self.init_weights)
        if args.is_load_pretrain==1:
            self._load_mae_all(args)
    def _load_mae_all(self,args):
        path = 'model/pretrain_' + args.dataset+'_num' + str(args.pretrain_num) + '_crop_size' + str(args.crop_size) + '_mask_ratio_' + str(args.mask_ratio)\
                             +'_DDH_' + str(args.depth)+str(args.dim)+str(args.head) +'.pth'
        state_dict = torch.load(path, map_location="cpu")['state_dict']
        ckpt_state_dict = {}
        for key, value in state_dict.items():
            if 'Encoder.' in key:
                if key[8:] in self.state_dict().keys():
                    ckpt_state_dict[key[8:]] = value
        for key, value in state_dict.items():
            if 'Decoder.' in key:
                if key[8:] in self.state_dict().keys():
                    ckpt_state_dict[key[8:]] = value

        for key, value in self.state_dict().items():
            if key not in ckpt_state_dict.keys():
                print('There only the FC have no load pretrain!!!', key)

        state = self.state_dict()
        state.update(ckpt_state_dict)
        self.load_state_dict(state)
        print("model load the mae!!!")
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x):
        # batch, c, h, w
        norm_embeeding_spa, sample_index_spa, mask_index_spa,norm_embeeding_chan, \
                        sample_index_chan, mask_index_chan = self.Encoder.autoencoder(x)
        proj_embeeding_spa = self.proj_spa(norm_embeeding_spa)
        proj_embeeding_chan = self.proj_chan(norm_embeeding_chan)
        decode_embeeding_spa,decode_embeeding_chan = self.Decoder.decoder(proj_embeeding_spa, sample_index_spa,
                                                mask_index_spa,proj_embeeding_chan,sample_index_chan, mask_index_chan)
        outputs_spa = self.restruction_spa(decode_embeeding_spa)
        outputs_chan = self.restruction_chan(decode_embeeding_chan)
        cls_token_spa = outputs_spa[:, 0, :]
        image_token_spa = outputs_spa[:, 1:, :] # (b, num_patches, patches_vector)
        # cal the mask patches normalization Independent
        image_norm_token_spa = self.patch_norm_spa(image_token_spa)
        n, l, dim = image_norm_token_spa.shape
        image_norm_token_spa = image_norm_token_spa.view(-1, self.num_patch[0], self.num_patch[1], dim).permute(0, 3, 1, 2)
        restore_image_spa = self.unconv_spa(image_norm_token_spa)

        cls_token_chan = outputs_chan[:, 0, :]
        image_token_chan = outputs_chan[:, 1:, :] # (b, num_patches, patches_vector)
        # cal the mask patches normalization Independent
        image_norm_token_chan = self.patch_norm_chan(image_token_chan)
        n, l, dim = image_norm_token_chan.shape
        image_norm_token_chan = self.unconv_chan(image_norm_token_chan)
        restore_image_chan = image_norm_token_chan.view(-1, self.num_patch[0], self.num_patch[1], l).permute(0, 3, 1, 2)
        return restore_image_spa, mask_index_spa,restore_image_chan, mask_index_chan


class VisionTransfromers(nn.Module):
    def __init__(self,
                 channel_number = 145,
                 img_size = 224,
                 patch_size = 16,
                 embed_dim = 192,
                 depth = 12,
                 num_heads = 3,
                 num_classes = 1000,
                 args = None
                 ):
        super(VisionTransfromers, self).__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.lidar_enc = lidar_e(args)
        self.hsi_enc = hsi_e(args)
        self.channel_num = channel_number
        self.conv = torch.nn.Conv2d(kernel_size=1, in_channels=128+64, stride=1, out_channels=self.img_size*self.img_size+2+self.channel_num)
        self.class_head = nn.Linear(self.img_size*self.img_size+2+self.channel_num, self.num_classes)
        base_cfg = dict(
            img_size=self.img_size,
            in_chans=channel_number,
            num_classes=self.num_classes,
            classification=True,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate = 0.,
            attn_drop_rate = 0.,
            drop_path_rate = 0.1,
            embed_layer_spa=PatchEmbed_spa,
            embed_layer_chan=PatchEmbed_chan,
            embed_dim = self.embed_dim,
            num_heads = self.num_heads,
            depth = self.depth,
            patch_size = self.patch_size,
            pos_embed="cosine",
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            pool='mean',
            args=args
        )

        self.model = VisionTransformer(**base_cfg)
        self.model.apply(self.init_weights)

        # if args.is_load_pretrain==1:
        #     self._load_mae_all(args)

    def forward(self, hsi,X,hsi_pca):
        x_spa,x_chan = self.model(torch.cat((hsi, X),1))
        e_hsi = self.hsi_enc(hsi_pca)
        e_X = self.lidar_enc(X)
        x = self.conv(torch.cat((e_hsi, e_X), 1))
        x = F.avg_pool2d(x, kernel_size=self.img_size).reshape(-1, self.img_size*self.img_size+2+self.channel_num)
        f = self.class_head(3*torch.cat((x_spa,x_chan),1)+x)
        return f, 3*torch.cat((x_spa,x_chan),1)+x

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def _load_mae_pretrain(self,args):
        path = 'model/pretrain_' + args.dataset+'_num' + str(args.pretrain_num) + '_crop_size' + str(args.crop_size) + '_mask_ratio_' + str(args.mask_ratio)\
                             +'_DDH_' + str(args.depth)+str(args.dim)+str(args.head) +'.pth'
        state_dict = torch.load(path, map_location="cpu")['state_dict']
        ckpt_state_dict = {}
        for key, value in state_dict.items():
            if 'Encoder.' in key:
                if key[8:] in self.model.state_dict().keys():
                    ckpt_state_dict[key[8:]] = value
        
        for key, value in self.model.state_dict().items():
            if key not in ckpt_state_dict.keys():
                print('There only the FC have no load pretrain!!!', key)
            
        state = self.model.state_dict()
        state.update(ckpt_state_dict)
        self.model.load_state_dict(state)
        print("model load the mae pretrain!!!")


    def _load_mae_all(self,args):
        path = 'model/pretrain_' + args.dataset+'_num' + str(args.pretrain_num) + '_crop_size' + str(args.crop_size) + '_mask_ratio_' + str(args.mask_ratio)\
                             +'_DDH_' + str(args.depth)+str(args.dim)+str(args.head) +'.pth'
        state_dict = torch.load(path, map_location="cpu")['state_dict']
        ckpt_state_dict = {}
        for key, value in state_dict.items():
            if 'Encoder.' in key:
                if key[8:] in self.model.state_dict().keys():
                    ckpt_state_dict[key[8:]] = value
        for key, value in state_dict.items():
            if 'Decoder.' in key:
                if key[8:] in self.model.state_dict().keys():
                    ckpt_state_dict[key[8:]] = value

        for key, value in self.model.state_dict().items():
            if key not in ckpt_state_dict.keys():
                print('There only the FC have no load pretrain!!!', key)
            
        state = self.model.state_dict()
        state.update(ckpt_state_dict)
        self.model.load_state_dict(state)
        print("model load the mae!!!")

if __name__ == '__main__':
    pass 
    