from typing import Any, Dict, Optional, Tuple, Union

import torch
from einops import rearrange

from .modules import CogVideoXBlock
from diffusers.models.attention import BasicTransformerBlock

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class ReferenceAttentionControl:
    def __init__(
        self,
        transformer,
        mode="write",
        reference_attn=True,
        reference_adain=False,
       
        dtype=torch.float16,
        device=torch.device("cpu"),
    ) -> None:
        
        self.transformer = transformer
        assert mode in ["read", "write"]
        
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
       
        self.register_reference_hooks(
            mode, 
            reference_attn,
            reference_adain,
            dtype=dtype,
        
            device=torch.device("cpu"),
           
           
        )

    def register_reference_hooks(
        self,
        mode,
        reference_attn,
        reference_adain,
        dtype=torch.float16,
        
        device=torch.device("cpu"),
        
    ):
        MODE = mode
       
       
        reference_attn = reference_attn
        reference_adain = reference_adain
        
       
        dtype = dtype
       

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            temb: torch.Tensor,
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            self_attention_additional_feats=None,
            mode=None,
            video_length=None,
        ):
           
            text_seq_length = encoder_hidden_states.size(1)

            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
                hidden_states, encoder_hidden_states, temb
            )
            
            if MODE == "write":
                
                self.bank.append(norm_hidden_states.clone())
                # attention
                attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                )

            if MODE == "read":
                
                bank_fea = [
                    # rearrange(
                    #     d.unsqueeze(1).repeat(1, video_length, 1, 1),
                    #     "b t l c -> (b t) l c",
                    # )
                    d
                    for d in self.bank
                ]
                
                modify_norm_encoder_hidden_states = torch.cat(
                    [norm_encoder_hidden_states] + bank_fea, dim=1
                )
                
                # modify_norm_encoder_hidden_states = torch.cat(
                #     [norm_encoder_hidden_states], dim=1
                # )
            
                attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=modify_norm_encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                    )
                attn_encoder_hidden_states = attn_encoder_hidden_states[:, :text_seq_length]

           
                hidden_states = hidden_states + gate_msa * attn_hidden_states
                encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

                # norm & modulate
                norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
                    hidden_states, encoder_hidden_states, temb
                )

                # feed-forward
                norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
                ff_output = self.ff(norm_hidden_states)

                hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
                encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

                return hidden_states, encoder_hidden_states

          
            hidden_states = hidden_states + gate_msa * attn_hidden_states
            encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
                hidden_states, encoder_hidden_states, temb
            )

            # feed-forward
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
            ff_output = self.ff(norm_hidden_states)

            hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

            return hidden_states, encoder_hidden_states



        if self.reference_attn:
            
            
            
            attn_modules = [module for module in torch_dfs(self.transformer) if isinstance(module, CogVideoXBlock)]
            
            # attn_modules = sorted(
            #     attn_modules, key=lambda x: -x.norm1.norm.normalized_shape[0]
            # )
            

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, CogVideoXBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, CogVideoXBlock
                    )
              

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer, dtype=torch.float16):
        
        if self.reference_attn: 
            
            reader_attn_modules = [
                module
                for module in torch_dfs(self.transformer)
                if isinstance(module, CogVideoXBlock)
            ]
            writer_attn_modules = [
                module
                for module in torch_dfs(writer.transformer)
                if isinstance(module, CogVideoXBlock)
            ]
           
            # import pdb; pdb.set_trace()
            # reader_attn_modules = sorted(
            #     reader_attn_modules, key=lambda x: -x.norm1.norm.normalized_shape[0]
            # )
            # writer_attn_modules = sorted(
            #     writer_attn_modules, key=lambda x: -x.norm1.norm.normalized_shape[0]
            # )
            
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r.bank = [v.clone().to(dtype) for v in w.bank]
                
                # w.bank.clear()

    def clear(self):
        if self.reference_attn:
           
         
            reader_attn_modules = [
                module
                for module in torch_dfs(self.transformer)
                if isinstance(module, CogVideoXBlock)
               
            ]
            # reader_attn_modules = sorted(
            #     reader_attn_modules, key=lambda x: -x.norm1.norm.normalized_shape[0]
            # )
            for r in reader_attn_modules:
                r.bank.clear()