import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from others.sam.segment_anything import sam_model_registry
from others.sam.sam_lora import LoRA_Sam


class SAMModel(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.image_embeddings = None

    def get_checkpoint(self, model_type):
        if model_type == "vit_b":
            checkpoint = os.path.join(self.opt.sam_weights_path, "sam_vit_b_01ec64.pth")
        elif model_type == "vit_l":
            checkpoint = os.path.join(self.opt.sam_weights_path, "sam_vit_l_0b3195.pth")
        elif model_type == "vit_h":
            checkpoint = os.path.join(self.opt.sam_weights_path, "sam_vit_h_4b8939.pth")
        else:
            raise ValueError("Model type error!")
        return checkpoint

    def setup(self):
        checkpoint = self.get_checkpoint(self.opt.sam_type)
        self.model = sam_model_registry[self.opt.sam_type](checkpoint=checkpoint)

        if self.opt.is_train:
            self.model.train()
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        self.finetune()

        return self

    def finetune(self):
        LoRA_Sam(self.model, self.opt.lora_rank)
        # self.set_norm_layer()
        # self.set_evp_adaptor_layer()
        # self.set_prompt_layer()

    def set_norm_layer(self):
        for name, param in self.model.image_encoder.named_parameters():
            if "norm" in name:
                param.requires_grad = True

    def set_evp_adaptor_layer(self):
        for param in self.model.image_encoder.prompt_generator.parameters():
            param.requires_grad = True

    def set_prompt_layer(self):
        self.model.image_encoder.Prompt_Tokens.requires_grad = True

    def reset_parameters(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "linear_a" in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                if "linear_b" in name:
                    nn.init.zeros_(param)

    def forward(self, images, prompts):
        _, _, H, W = images.shape
        image_embeddings = self.encode(images)
        pred_masks, ious, res_masks = self.decode((H, W), prompts)
        return image_embeddings, pred_masks, ious, res_masks

    def encode(self, images):
        self.image_embeddings = self.model.image_encoder(images)
        return self.image_embeddings

    def decode(self, image_shape, prompts):
        image_embeddings = self.image_embeddings
        if image_embeddings is None:
            raise "No image embeddings"

        pred_masks = []
        ious = []
        res_masks = []

        prompt_coordinates, prompt_labels = prompts
        prompts_list = []
        for b in range(prompt_coordinates.shape[0]):
            prompts_list.append([prompt_coordinates[b].unsqueeze(0), prompt_labels[b].unsqueeze(0)])

        for prompt, embedding in zip(prompts_list, image_embeddings):
            if isinstance(prompt, torch.Tensor):
                prompt = prompt.to(device=embedding.device)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=prompt,
                masks=None,
            )
            elif isinstance(prompt, (tuple, list)):
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=prompt,
                boxes=None,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                image_shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)

        pred_masks = torch.stack(pred_masks, dim=0)
        ious = torch.stack(ious, dim=0)
        res_masks = torch.stack(res_masks, dim=0)

        return pred_masks, ious, res_masks
