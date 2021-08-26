import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedd(nn.Module):

    def __init__(self, vocab_size, d_model, max_len = 50):
        super(Embedd, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.positinal_encoding(max_len, self.d_model)
        
    def  positinal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):  
            for i in range(0, d_model, 2): #from zero to d_model , with 2 steps   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)   # from  [max_len,d_model] to  [1,max_len,d_model]
        return pe
        
    def forward(self, enc_words):
        embedd = self.embed(enc_words) * math.sqrt(self.d_model)
        embedd += self.pe[:, :embedd.size(1)]   
        embedd = self.dropout(embedd)
        return embedd



class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)
    def project_reshape(self,x):
         # (batch_size, max_len, d_model) -> (batch_size, max_len, h, d_k) -> (batch_size, h, max_len, d_k)
         x = x.view(x.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
         return x
    def attention(self,query, key, value , mask):
        # dot prodects between query and the transpose of key , with normalzation
        scores = torch.matmul(query, key.permute(0,1,3,2)) / math.sqrt(query.size(-1))
        scores = scores.masked_fill(mask == 0, -1e9)  # mask the zeros of padding
        weights = F.softmax(scores, dim = -1)   
        weights = self.dropout(weights)
        attended = torch.matmul(weights, value)
        return attended
    def conacats(self, x):
        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, h * d_k)
        x = x.permute(0,2,1,3).contiguous().view(x.shape[0], -1, self.heads * self.d_k)
        return self.concat(x)
    def forward(self, query, key, value, mask):
        query = self.project_reshape(self.query(query))  
        key = self.project_reshape(self.key(key))        
        value = self.project_reshape(self.value(value))
        contexts = self.attention(query, key, value , mask) 
        out = self.conacats(contexts)  
        return out 



class FeedForward(nn.Module):

    def __init__(self, d_model, middle_dim = 2048):
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, Embedd, mask):
        interacted = self.dropout(self.self_multihead(Embedd, Embedd, Embedd, mask))
        interacted = self.layernorm(interacted + Embedd)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded


class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.src_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, Embedd, encoded, src_mask, target_mask):
        query = self.dropout(self.self_multihead(Embedd, Embedd, Embedd, target_mask))
        query = self.layernorm(query + Embedd)
        interacted = self.dropout(self.src_multihead(query, encoded, encoded, src_mask))
        interacted = self.layernorm(interacted + query)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        decoded = self.layernorm(feed_forward_out + interacted)
        return decoded


class Transformer(nn.Module):
    
    def __init__(self, d_model, heads, num_layers, word_map):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = len(word_map)
        self.embed = Embedd(self.vocab_size, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(num_layers)])
        self.logit = nn.Linear(d_model, self.vocab_size)
        
    def encode(self, src_words, src_mask):
        src_Embedd = self.embed(src_words)
        for layer in self.encoder:
            src_Embedd = layer(src_Embedd, src_mask)
        return src_Embedd
    
    def decode(self, target_words, target_mask, src_Embedd, src_mask):
        tgt_Embedd = self.embed(target_words)
        for layer in self.decoder:
            tgt_Embedd = layer(tgt_Embedd, src_Embedd, src_mask, target_mask)
        return tgt_Embedd
        
    def forward(self, src_words, src_mask, target_words, target_mask):
        encoded = self.encode(src_words, src_mask)
        decoded = self.decode(target_words, target_mask, encoded, src_mask)
        out = F.log_softmax(self.logit(decoded), dim = 2)
        return out