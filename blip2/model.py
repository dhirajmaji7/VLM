import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import timm
from transformers import BertConfig, BertModel
from transformers import AutoModelForSeq2SeqLM


class BertMLPBlock(nn.Module):
    def __init__(self, intermediate, output):
        super().__init__()
        self.intermediate = intermediate
        self.output = output

    def forward(self, x):
        intermediate_output = self.intermediate(x)
        return self.output(intermediate_output, x)
    

class BertEncoderBlock(nn.Module):
    def __init__(self, bert_layer, bert_config, is_cross_attn=False):
        super().__init__()
        self.bert_config = bert_config
        self.is_cross_attn = is_cross_attn
        self.self_attn = bert_layer.attention
        self.mlp_img_transformer = BertMLPBlock(bert_layer.intermediate, bert_layer.output)
        self.mlp_text_transformer = BertMLPBlock(
                    copy.deepcopy(bert_layer.intermediate), 
                    copy.deepcopy(bert_layer.output)
                    )
        if is_cross_attn:
            self.cross_attn = nn.MultiheadAttention(embed_dim=self.bert_config.hidden_size, 
                                                    num_heads=self.bert_config.num_attention_heads, 
                                                    batch_first=True)
            self.cross_layer_norm = nn.LayerNorm(self.bert_config.hidden_size)
        
    def forward(self, query_embds, img_embds, text_embds, attn_mask):
        _, Qs, _ = query_embds.shape
        _, Ts, _ = text_embds.shape

        combined_embds = torch.concat((query_embds, text_embds), dim=1) # B, Qs + Ts, D

        self_attn_output = self.self_attn(combined_embds, attention_mask=attn_mask)[0]
        query_embds = combined_embds[:, :Qs]
        text_embds= combined_embds[:, Qs:]
        
        if self.is_cross_attn:
            hidden_states = self.cross_attn(query_embds, img_embds, img_embds)[0]
            query_embds = self.cross_layer_norm(query_embds + hidden_states)

        query_embds = self.mlp_img_transformer(query_embds)
        text_embds = self.mlp_text_transformer(text_embds)
        return query_embds, text_embds


class QTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_cfg  = BertConfig.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", config = self.bert_cfg)
        
        self.encoder = nn.ModuleList()
        for i, bert_layer in enumerate(self.bert_model.encoder.layer):
            self.encoder.append(BertEncoderBlock(bert_layer, self.bert_cfg, i % 2 == 0))
        
        qs = config.num_queries
        ts = config.context_length
        combined_seq_len = qs + ts

        ####  STAGE 1: ITC, ITM, ITG Loss Masks ####
        # ITC Loss Mask
        itc_attn_mask = torch.zeros((combined_seq_len, combined_seq_len))
        itc_attn_mask[:qs, :qs] = 1
        itc_attn_mask[qs:, qs:] = 1

        # ITM Loss Mask
        itm_attn_mask = torch.ones((combined_seq_len, combined_seq_len))

        # ITG Loss Mask
        itg_attn_mask = torch.ones((combined_seq_len, combined_seq_len))
        itg_attn_mask[:qs, qs:] = 0
        itg_attn_mask[qs:, qs:] = torch.tril(itg_attn_mask[qs:, qs:], diagonal=0)

        self.register_buffer("itc_attn_mask", itc_attn_mask)
        self.register_buffer("itm_attn_mask", itm_attn_mask)
        self.register_buffer("itg_attn_mask", itg_attn_mask)

        ####  STAGE 2: ####
        # ITC Loss Mask will be same as stage 1 and reused for stage 2

    def forward(self, query_embds, img_embds, cls_text_embds, dec_text_embds, stage):

        itc_query_embds = query_embds.clone()
        itm_query_embds = query_embds.clone()
        itg_query_embds = query_embds.clone()

        itc_text_embds = cls_text_embds.clone()
        itm_text_embds = cls_text_embds.clone()
        itg_text_embds = dec_text_embds.clone()


        for encoder in self.encoder:
            itc_query_embds, itc_text_embds = encoder(itc_query_embds, img_embds, itc_text_embds, self.itc_attn_mask)
            if stage == 1:
                itm_query_embds, itm_text_embds = encoder(itm_query_embds, img_embds, itm_text_embds, self.itm_attn_mask)
                itg_query_embds, itg_text_embds = encoder(itg_query_embds, img_embds, itg_text_embds, self.itg_attn_mask)
        return itc_query_embds, itc_text_embds, itm_query_embds, itm_text_embds, itg_query_embds, itg_text_embds
    

class QFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q_transformer = QTransformer(config)
        self.learned_query = nn.Parameter(torch.randn(config.num_queries, config.embedding_dim))
        self.output_embedding  = nn.Embedding(config.bert_vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)

        position_ids = torch.arange(self.config.context_length).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

    def forward(self, image_embedding: torch.tensor, cls_tokens: torch.tensor, dec_tokens: torch.tensor, stage:int):
        B, S, E = image_embedding.shape
        learned_query = self.learned_query.unsqueeze(0).expand(B, -1, -1)

        cls_text_embeddings = self.output_embedding(cls_tokens) #(S,768)
        cls_text_embeddings = cls_text_embeddings + self.position_embedding(self.position_ids.expand(B, -1))
        dec_text_embeddings = self.output_embedding(dec_tokens) #(S,768)
        dec_text_embeddings = dec_text_embeddings + self.position_embedding(self.position_ids.expand(B, -1))

        itc_query_embds, itc_text_embds, itm_query_embds, itm_text_embds, itg_query_embds, itg_text_embds = self.q_transformer(
            learned_query, image_embedding, cls_text_embeddings, dec_text_embeddings, stage)

        if itg_text_embds is not None:
            itg_logits = itg_text_embds @ self.output_embedding.weight.T # (S,Vocab_size)
        else:
            itg_logits = None

        return itc_query_embds, itc_text_embds, itm_query_embds, itm_text_embds, itg_logits


class FlanT5Model(nn.Module):
    def __init__(self):
        super(FlanT5Model, self).__init__()
        self.lm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        for param in self.lm_model.parameters():
            param.requires_grad = False

    def forward(self, query_embedding, input_token, label, enc_mask):
        #query_embd : (B,32,512)
        # input_token : (B,L)
        B, Q, d = query_embedding.shape
        device = query_embedding.device
        with torch.no_grad():
            input_embd = self.lm_model.encoder.embed_tokens(input_token)  #(B,L,512)

        encoder_input = torch.concat((query_embedding, input_embd) , dim = 1).contiguous()

        prefix_mask = torch.ones((B, Q ), dtype= enc_mask.dtype, device=device)
        attention_mask = torch.concat((prefix_mask, enc_mask) , dim=1).contiguous()  # [B, 32+L]
        label = label.contiguous()  # [B, L]
        out = self.lm_model(inputs_embeds=encoder_input,
                                attention_mask=attention_mask,
                                labels=label,
                                return_dict=True)
        return out
    

    def predict(self, query_embedding, input_token, enc_mask):
        B, Q, d = query_embedding.shape
        device = query_embedding.device
        with torch.no_grad():
            input_embd = self.lm_model.encoder.embed_tokens(input_token)  #(B,L,512)

        encoder_input = torch.concat((query_embedding, input_embd) , dim = 1)

        prefix_mask = torch.ones((B, Q ), dtype= enc_mask.dtype, device=device)
        attention_mask = torch.concat((prefix_mask, enc_mask) , dim=1)  # [B, 32+L]
        
        enc_out = self.lm_model.encoder(
            inputs_embeds=encoder_input,
            attention_mask=attention_mask,
            return_dict=True
            )

        gen_ids = self.lm_model.generate(
            encoder_outputs=enc_out,
            max_new_tokens=30,
            decoder_start_token_id=self.lm_model.config.decoder_start_token_id,
            attention_mask=attention_mask,
        )

        return gen_ids


class Blip2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.image_encoder = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.image_encoder.reset_classifier(0)

        for param in self.parameters():
            param.requires_grad = False

        self.image_proj = nn.Linear(config.img_embd_dim, config.embedding_dim)

        self.q_former = QFormer(config)
        self.z_proj = nn.Linear(config.embedding_dim, config.lm_embedding_dim)

        self.lm_model = FlanT5Model()
    
    
    def stage1(self, image:torch.tensor, cls_caption:torch.tensor, dec_caption:torch.tensor):
        image_embedding = self.image_encoder.forward_features(image)  # [B, C, F]
        image_embedding = self.image_proj(image_embedding)

        itc_query_embds, itc_text_embds, itm_query_embds, itm_text_embds, itg_logits = self.q_former(image_embedding, cls_caption, dec_caption, 1)
        return itc_query_embds, itc_text_embds, itm_query_embds, itm_text_embds, itg_logits
    
    
    def stage2(self, image, input_token, label, enc_mask, dummy_input_size):
        image_embedding = self.image_encoder.forward_features(image)  # [B, C, F]
        image_embedding = self.image_proj(image_embedding)
        
        cls_caption_dummy = torch.zeros(dummy_input_size, dtype=torch.long, device = image.device)
        dec_caption_dummy = torch.zeros(dummy_input_size, dtype=torch.long, device = image.device)
        itc_query_embds, itc_text_embds, itm_query_embds, itm_text_embds, itg_logits = self.q_former(image_embedding, 
                                                            cls_caption_dummy, dec_caption_dummy, 2)
        
        z = self.z_proj(itc_query_embds)  # [B, Qs, D]

        out = self.lm_model(z, input_token, label, enc_mask)
            
        return out