#***********************************Cute Author********************************************** 
#★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★★⋅☆⋅★
#  ██╗   ██╗ █████╗ ███████╗██╗███╗   ██╗   █████╗ ██████╗  █████╗ ███████╗ █████╗ ████████╗
#  ██║   ██║██╔══██╗██╔════╝██║████╗  ██║  ██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗╚══██╔══╝
#  ██║   ██║███████║███████╗██║██╔██╗ ██║  ███████║██████╔╝███████║█████╗  ███████║   ██║   
#  ╚██╗ ██╔╝██╔══██║╚════██║██║██║╚██╗██║  ██╔══██║██╔══██╗██╔══██║██╔══╝  ██╔══██║   ██║   
#   ╚████╔╝ ██║  ██║███████║██║██║ ╚████║  ██║  ██║██║  ██║██║  ██║██║     ██║  ██║   ██║   
#    ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝   ╚═╝   
# ★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★⋅☆⋅★★⋅☆⋅
# *******************************************************************************************
# Email: yasinarafat.e2021@gmail.com
# Starting Day date: 19-05-25
# Last Modificaion:  25-5-25


import math
import torch
import numpy as np 
import torch.nn as nn 
from  torch.nn import functional as F 

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 


# ******************************************************************
# *********************Positional Encoding:*************************
# ******************************************************************

class PostionalEncoding(nn.Module):
    def __init__(self,d_model,max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        
    def forward(self): #let's d_model=512 and max_sequence_length=200
        # in our note it will be matrix row:
        # [embeding]
        # [embeding] max_sequence length X embeding 
        # that why we reshape it (1 X max_seq_length) to (max_seq_length X 1)
        pos = torch.arange(start=0,end=self.max_sequence_length,step=1).reshape(
            self.max_sequence_length,1) # 200X1
        # i = 0 to d_model/2
        # 2i = 0,2,4,..d_model
        two_i = torch.arange(start=0,end=self.d_model,step=2).float() # 1X(512/2) = 1x256
        denominator = torch.pow(input=torch.tensor(10000),exponent=(two_i/self.d_model)) # 1x256
         # (200x1)/(1x256) = 200x256 (PyTorch automatically expands dimention for this operation)
        even_pos = torch.sin(pos/denominator) # 200x256
        odd_pos = torch.cos(pos/denominator) # 200x256 
        stacked = torch.stack(tensors=(even_pos,odd_pos),dim=2) # 200x256x2
        PE = torch.flatten(dims=1) #200x(256x2) = #200x512
        return PE 
    

# ******************************************************************
# *********************Sentence Embedding:*************************
# ******************************************************************
"""For a given sentence it will find embedding -> use"nn.Embedding" """
class SentenceEmbedding(nn.Module):
    def __init__(self,max_sequence_length,d_model,language_to_index,START_TOKEN,
                 END_TOKEN,PADDING_TOKEN):
        super().__init__()
        self.d_model = d_model
        self.END_TOKEN = END_TOKEN
        self.START_TOKEN = START_TOKEN
        self.dropOut = nn.Dropout(p=0.1)
        self.PADDING_TOKEN = PADDING_TOKEN
        self.vocab_size = len(language_to_index)
        self.language_to_index = language_to_index
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size,d_model)
        self.positional_encoding = PostionalEncoding(d_model=d_model,
                                        max_sequence_length=max_sequence_length)
        
    def batch_tokenizer(self,batch,start_token,end_token):
        # for a single sentence
        def tokenize(sentence,start_token,end_token):
            sentence_word_indexies = [ self.language_to_index for i in list(sentence)]
            # start of sentence
            if start_token:
                sentence_word_indexies.insert(0,self.language_to_index[self.START_TOKEN])
            # end of sentence
            if end_token:
                sentence_word_indexies.append(self.language_to_index[self.END_TOKEN])
            # padding
            for _ in range(len(sentence_word_indexies),self.max_sequence_length):
                sentence_word_indexies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indexies)
        
        # for a batch of sentences
        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(tokenize(sentence=batch[sentence_num],start_token=
                                      start_token,end_token=end_token))
        tokenized = torch.stack(tokenized)
        # as we have batch, we can use the power of gpu
        return tokenized.to(get_device())
    
    def forward(self,x,start_token,end_token):
        x = self.batch_tokenizer(x,start_token,end_token)
        x = self.embedding(x)
        pos = self.positional_encoding().to(get_device())
        # PE not learnable but x embedding is learnable
        # The below operation will droput after addition 
        # Why dropout: to learn embedding correctly 
        # droping nn: will handle unseen token during inference
        x = self.dropOut(x+pos)
        return x


# ******************************************************************
# *********************Multi head attention:************************
# ******************************************************************

# multi_head_attention = attention + attention + attention + ... 
# assume batch_size = 30, num_heads = 8 
def attention(Q,K,V,mask=None):
    #30 X 8 X 200 X 64 (batch, heads, max_seq_len, d_k)
    d_k = K.size()[1]
    #30 x 8 x 64 x 200,(30->batch,8->head) not require to change
    matmul = torch.matmul(input=Q,other=K.transpose(-1,-2))/math.sqrt(d_k) # 30x8x200x200
    if mask is not None:
        matmul += mask # pytoch automatically determine shpae and add mask(200X200)
    # 64 dimention of embedding, max_seq_lenth = num of total token
    # output: which token probability is high, so we ne to apply 
    # softmax at max_seq_length dimention
    attn_score = F.softmax(input=matmul,dim=-1) # 30x8x200x200
    attn_weight_val = torch.matmul(input=V,other=attn_score) #30x8x200x64
    return attn_weight_val #30x8x200x64
    
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,d_model):
        super().__init__()
        self.num_heads = num_heads #8
        self.d_model = d_model #512
        self.head_dim = d_model // num_heads #64
        self.qkv = nn.Linear(in_features=d_model,out_features=3*d_model) #512x1524
        self.linear_transfm = nn.Linear(in_features=d_model,out_features=d_model) #512x512
        
    def forward(self,x,mask): 
        batch_size,max_sequence_length,d_model = x.size() # 30x200x512
        qkv = self.qkv #512x1524
        qkv = qkv.view(batch_size,max_sequence_length,
                       self.num_heads,3*self.head_dim)#30x200x8x192
        qkv = qkv.permute(0,2,1,3)#30x8x200x192
        q,k,v = qkv.chunk(3,dim=-1) #30x8x200x64,30x8x200x64,30x8x200x64
        attn_weight_val = attention(Q=q,K=k,V=v,mask=mask) # 30x8x200x64
        attn_weight_val = attn_weight_val.permute(0,2,1,3).reshape(batch_size,
                                                                   max_sequence_length,
                                                                   self.num_heads*self.head_dim
                                                                   )#30x200x512
        # send concatenate result into final liner transformation layer
        out = self.linear_transfm(attn_weight_val)
        return out
    
    
# We don't need masked_multi_head_attention cause, we add a mask property.
# if we don't use mask then it's become multi_head_attention
# if we use mask then then it's become masked_multi_head_attention


# ******************************************************************
# *********************Cross attention:*****************************
# ******************************************************************
class CrossAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.num_heads = num_heads #8
        self.d_model = d_model #512
        self.head_dim = d_model // num_heads #64
        self.kv = nn.Linear(in_features=d_model,out_features=2*d_model) #512x1024
        self.q = nn.Linear(in_features=d_model,out_features=d_model) #512x512
        self.linear_transfm = nn.Linear(in_features=d_model,out_features=d_model) #512x512
        
    def forward(self,x,y,mask):
        batch_size,max_sequence_length,d_model = x.size() # 30x200x512
        kv = self.kv(x) #512x1024
        q = self.v(y) #512x512
        kv = kv.view(batch_size,max_sequence_length,
                       self.num_heads,2*self.head_dim)#30x200x8x128
        q = q.view(batch_size,max_sequence_length,
                       self.num_heads,2*self.head_dim)#30x200x8x64
        
        kv = kv.permute(0,2,1,3)#30x8x200x128
        q= q.permute(0,2,1,3)#30x8x200x64
        
        k,v = kv.chunk(2,dim=-1) #30x8x200x64,30x8x200x64
        
        attn_weight_val = attention(Q=q,K=k,V=v,mask=mask) # 30x8x200x64
        attn_weight_val = attn_weight_val.permute(0,2,1,3).reshape(batch_size,
                                                                   max_sequence_length,
                                                                   self.num_heads*self.head_dim
                                                                   )#30x200x512
        # send concatenate result into final liner transformation layer
        out = self.linear_transfm(attn_weight_val)
        return out
    
        

# ******************************************************************
# ********************* Layer Normalization ************************
# ******************************************************************
class LayerNormalization(nn.Module):
    def __init__(self,parameters_shape,esp=1e-5):
        super().__init__()   
        self.parameters_shape = parameters_shape
        self.beta = nn.parameter(torch.zeros(size=parameters_shape))
        self.gamma = nn.Parameter(torch.ones(size=parameters_shape))
        self.esp = esp 
    def forward(self,x):
        dims = [-(i+1) for i in range(len(self.parameters_shape))] #512
        mean = x.mean(dim=dims,keepdim=True)
        var = ((x-mean)**2).mean(dim=dims,keepdim=True)
        std = (var+self.esp).sqrt()
        y = (x-mean)/std
        output = y * self.gamma + self.beta
        return output
    
# ******************************************************************
# ****************Feed Forward Network:*****************************
# ******************************************************************
class FeedForward(nn.Module):
    def __init__(self,d_model,hidden,drop_prob):
        super().__init__()
        self.linear1 = nn.Linear(in_features=d_model,out_features=hidden)
        self.linear2 = nn.Linear(in_features=hidden,out_features=d_model)
        self.dropOut = nn.Dropout(p=drop_prob)
        self.activation = nn.ReLU()
    def forward(self,x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropOut(x)
        x = self.linear2(x)
        return x 
    
# ******************************************************************
# ***********************Encoder Layer *****************************
# ******************************************************************
class EncoderLayer(nn.Module):
    def __init__(self,num_heads,hidden,d_model,drop_prob):
        super().__init__()
        self.multihead = MultiHeadAttention(num_heads=num_heads,d_model=d_model)
        self.dropOut1 = nn.Dropout(p=drop_prob)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.ffn = FeedForward(d_model=d_model,hidden=hidden,drop_prob=drop_prob)
        self.dropOut2 = nn.Dropout(p=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
    def forward(self,x,self_attention_mask):
        residual_x = x
        x = self.multihead(x,self_attention_mask)
        x = self.dropOut1(x)
        x = self.norm1(residual_x + x)
        
        residual_x = x 
        x = self.ffn(x)
        x = self.dropOut2(x)
        x = self.norm2(x + residual_x)
        return x 
    

    
# ******************************************************************
# ***********************Encoder Block *****************************
# ******************************************************************

# Sequentially apply a series of encoder layers
# a sequence of modules (layers) in order, passing the output of 
# one module as the input to the next, More in encoder.ipynb

class SequentialEncoder(nn.Sequential):
    def forward(self,*inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x,self_attention_mask)
 


class Encoder(nn.Module):
    # cpy-paste from EncoderLayer and SentenceEmbedding:
    def __init__(self,
                 num_heads,
                 hidden,
                 d_model,
                 drop_prob,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN,
                 num_layers
                 ):
        super().__init__()
        self.embedding = SentenceEmbedding(max_sequence_length,d_model,
                                           language_to_index,START_TOKEN,END_TOKEN,
                                           PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(num_heads,hidden,d_model,drop_prob) 
                                          for _ in range(num_layers)])
    
    def forward(self,x,self_attention_mask,start_token,end_token):
        # forward method of SentenceEmbedding:
        x = self.embedding(x,start_token,end_token)
        # forward method of EncoderLayer
        x = self.layers(x,self_attention_mask)
    

    
# ******************************************************************
# ***********************Decoder Layer *****************************
# ******************************************************************

class DecoderLayer(nn.Module):
    def __init__(self,num_heads,d_model,hidden,drop_prob):
        super().__init__()
        self.maskedattention = MultiHeadAttention(num_heads,d_model)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.drop1 = nn.Dropout(p=drop_prob)
        
        self.crossattention = CrossAttention(d_model,num_heads) 
        self.norm2 = LayerNormalization([d_model])
        self.drop2 = nn.Dropout(p=drop_prob)
        
        self.ffn = FeedForward(d_model,hidden,drop_prob)
        self.norm3 = LayerNormalization([d_model])
        self.drop3 = nn.Dropout(p=drop_prob)
        
    def forward(self,x,y,self_attention_mask,cross_attention_mask):
        _yResidual = y 
        y = self.maskedattention(y,self_attention_mask)
        y = self.drop1(y)
        y = self.norm1(_yResidual+y)
        
        _yResidual = y 
        y = self.crossattention(x,y,cross_attention_mask)
        y = self.drop2(y)
        y = self.norm2(_yResidual+y)
        
        _yResidual = y 
        y = self.ffn(y)
        y = self.drop3(y)
        y = self.norm3(_yResidual+y)
        return y 
    

    
# ******************************************************************
# ***********************Decoder Block *****************************
# ******************************************************************

class SequentialDecoder(nn.Sequential):
    def forward(self,*inputs):
        x,y,self_attention_mask,cross_attention_mask = input
        for module in self._modules.values():
            y = module(x,y,self_attention_mask,cross_attention_mask)

class Decoder(nn.Module):
    def __init__(self,
                 max_sequence_length,
                 d_model,language_to_index,
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN,
                 num_heads, hidden,
                 drop_prob,num_layers):
        self.embedding = SentenceEmbedding(max_sequence_length,d_model,
                                                   language_to_index,START_TOKEN,END_TOKEN,
                                                   PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(num_heads,d_model,hidden,drop_prob) 
                                          for _ in range(num_layers)])
        super().__init__()
    
    def forward(self,x,y,self_attention_mask,cross_attention_mask,start_token,end_token):
        y = self.embedding(y,start_token,end_token)
        y = self.layers(x,y,self_attention_mask,cross_attention_mask)
        return y 
    
    

# ******************************************************************
# ********************Transformer Block ****************************
# ******************************************************************
  
class Transformer(nn.Module):
    def __init__(self,
                 max_sequence_length,
                 d_model,
                 english_to_index,
                 bangla_to_index,
                 bangla_vocab_size,
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN,
                 num_heads,
                 hidden,
                 drop_prob,
                 num_layers):
        super().__init__()
        self.encoder = Encoder(num_heads,hidden,d_model,drop_prob,
                               max_sequence_length,english_to_index,
                               START_TOKEN,END_TOKEN,PADDING_TOKEN,num_layers)
        self.decoder = Decoder(max_sequence_length,d_model,
                               bangla_to_index,START_TOKEN,END_TOKEN,
                               PADDING_TOKEN,num_heads,
                               hidden,drop_prob,num_layers)
        
        self.linear = nn.Linear(d_model,bangla_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    def forward(self,
                x,y,
                encoder_self_attention_mask=None,
                decoder_self_attention_mask=None,
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=True):
        x = self.encoder(x,encoder_self_attention_mask,enc_start_token,enc_end_token)
        out = self.decoder(x,y,decoder_self_attention_mask,
                           decoder_cross_attention_mask,
                           dec_start_token,
                           dec_end_token)
        out = self.linear(out)
        return out 
    
        

