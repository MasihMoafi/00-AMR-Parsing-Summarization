# CELL 1: COMMON SETUP, DATA LOADING, AMR PARSING, TOKENIZATION & VOCAB, DATASET & DATALOADER
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import amrlib

# SET CUDA 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DATA LOADING
def load_data(path):
    df = pd.read_csv(path)
    articles = df['article'].tolist()
    highlights = df['highlights'].tolist()
    return articles, highlights

train_articles, train_highlights = load_data("/home/masih/Downloads/Telegram Desktop/data/train.csv")  # REPLACE WITH YOUR PATH
val_articles, val_highlights = load_data("/home/masih/Downloads/Telegram Desktop/data/validation.csv")
test_articles, test_highlights = load_data("/home/masih/Downloads/Telegram Desktop/data/test.csv")

# AMR PARSING & PREPROCESSING 
stog = amrlib.load_stog_model(device='cpu')

def parse_amr(articles):
    print("Parsing AMR graphs...")
    amr_graphs = stog.parse_sents(articles)
    return [g if g else "" for g in amr_graphs]

# Process all splits
train_graphs = parse_amr(train_articles)
val_graphs = parse_amr(val_articles)
test_graphs = parse_amr(test_articles)

# TOKENIZATION & VOCAB 
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>"}
        
    def build_vocab(self, texts, max_size=2000):
        words = [word for text in texts for word in text.split()]
        word_counts = Counter(words)
        common_words = word_counts.most_common(max_size)
        
        for idx, (word, _) in enumerate(common_words, start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
VOCAB_SIZE = 2000
vocab = Vocabulary()
vocab.build_vocab(train_graphs + train_highlights, max_size=VOCAB_SIZE)

# DATASET & DATALOADER 
class SummaryDataset(Dataset):
    def __init__(self, graph_strings, highlights, vocab):
        self.graphs = [self.text_to_ids(gs, vocab) for gs in graph_strings]
        self.highlights = [self.text_to_ids(s, vocab, add_special=True) for s in highlights]
        
    def text_to_ids(self, text, vocab, add_special=False):
        ids = [vocab.word2idx.get(word, 1) for word in text.split()]
        if add_special:
            ids = [vocab.word2idx["<sos>"]] + ids + [vocab.word2idx["<eos>"]]
        return ids
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.graphs[idx]),
            torch.tensor(self.highlights[idx])
        )

def collate_fn(batch):
    srcs, trgs = zip(*batch)
    srcs = torch.nn.utils.rnn.pad_sequence(srcs, padding_value=0).transpose(0, 1)
    trgs = torch.nn.utils.rnn.pad_sequence(trgs, padding_value=0).transpose(0, 1)
    return srcs, trgs

# CREATE DATASETS AND DATALOADERS
train_dataset = SummaryDataset(train_graphs, train_highlights, vocab)
val_dataset = SummaryDataset(val_graphs, val_highlights, vocab)
test_dataset = SummaryDataset(test_graphs, test_highlights, vocab)

train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)


# CELL 2: AS2SP Model 
import math

class AS2SP(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.enc_embed = nn.Embedding(vocab_size, 128)
        self.encoder = nn.LSTM(128, 64, 
                               num_layers=1,
                               bidirectional=True,
                               batch_first=True)
        self.hidden_proj = nn.Linear(64 * 2, 256)
        self.cell_proj = nn.Linear(64 * 2, 256)
        self.dec_embed = nn.Embedding(vocab_size, 128)
        self.decoder = nn.LSTM(128, 256, num_layers=1, batch_first=True)
        self.W_h = nn.Linear(64 * 2, 256)
        self.W_s = nn.Linear(256, 256)
        self.v = nn.Linear(256, 1)
        self.p_gen = nn.Linear(128 + 256 + 128, 1)
        self.fc = nn.Linear(256, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, src_graph, trg_text):
        enc_embedded = self.dropout(self.enc_embed(src_graph))
        enc_out, (h_n, c_n) = self.encoder(enc_embedded)
        
        h_n = torch.cat([h_n[0], h_n[1]], dim=-1)
        c_n = torch.cat([c_n[0], c_n[1]], dim=-1)
        
        decoder_hidden = self.hidden_proj(h_n).unsqueeze(0)
        decoder_cell = self.cell_proj(c_n).unsqueeze(0)
        
        dec_embedded = self.dropout(self.dec_embed(trg_text))
        dec_out, _ = self.decoder(dec_embedded, (decoder_hidden, decoder_cell))
        
        enc_proj = self.W_h(enc_out).unsqueeze(2)
        dec_proj = self.W_s(dec_out).unsqueeze(1)
        
        attn_energy = torch.tanh(enc_proj + dec_proj)
        attn_scores = self.v(attn_energy).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = attn_weights.permute(0, 2, 1)
        context = torch.bmm(attn_weights, enc_out)
        
        p_gen_input = torch.cat([context, dec_out, dec_embedded], dim=-1)
        p_gen = torch.sigmoid(self.p_gen(p_gen_input))
        
        output = self.fc(dec_out)
        return output, attn_weights, p_gen

# TRAINING SETUP 
model = AS2SP(VOCAB_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(3):
    model.train()
    total_loss = 0
    for batch_idx, (src, trg) in enumerate(train_loader):
        src, trg = src.to(device), trg.to(device)
        
        outputs, _, _ = model(src[:, :-1], trg[:, :-1])
        loss = criterion(outputs.reshape(-1, VOCAB_SIZE), 
                         trg[:, 1:].reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.4f}")

print("Training completed!")

def generate_summary(model, graph_string, vocab, max_len=20):
    model.eval()
    tokenized = [vocab.word2idx.get(word, 1) for word in graph_string.split()]
    src = torch.tensor([tokenized]).to(device)
    
    decoder_input = torch.tensor([[vocab.word2idx["<sos>"]]]).to(device)
    summary = []
    
    with torch.no_grad():
        enc_embedded = model.enc_embed(src)
        enc_out, (h_n, c_n) = model.encoder(enc_embedded)
        
        h_n = torch.cat([h_n[0], h_n[1]], dim=-1)
        c_n = torch.cat([c_n[0], c_n[1]], dim=-1)
        decoder_hidden = model.hidden_proj(h_n).unsqueeze(0)
        decoder_cell = model.cell_proj(c_n).unsqueeze(0)
        
        for _ in range(max_len):
            dec_embedded = model.dec_embed(decoder_input)
            dec_out, (decoder_hidden, decoder_cell) = model.decoder(
                dec_embedded, (decoder_hidden, decoder_cell)
            )
            
            output = model.fc(dec_out)
            next_token = output.argmax(-1)[:, -1].item()
            
            if next_token == vocab.word2idx["<eos>"]:
                break
            
            summary.append(vocab.idx2word.get(next_token, "<unk>"))
            decoder_input = torch.tensor([[next_token]]).to(device)
            
    return " ".join(summary)

print("\nGenerated Summaries for Test Set:")
for i in range(len(test_dataset)):
    input_graph = test_graphs[i]
    generated = generate_summary(model, input_graph, vocab)
    print(f"Original Article: {test_articles[i]}")
    print(f"Generated Summary: {generated}")
    print(f"Reference Summary: {test_highlights[i]}\n{'-'*50}")


# CELL 3: TRCE MODEL (TRANSFORMER WITH CONTEXTUAL EMBEDDINGS)
from transformers import BertTokenizer, BertModel

class TRCEModel(nn.Module):
    def __init__(self, vocab_size, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_emb_size = 768 
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.transformer = nn.Transformer(
            d_model=self.bert_emb_size,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        
        self.dec_embed = nn.Embedding(vocab_size, self.bert_emb_size)
        self.fc_out = nn.Linear(self.bert_emb_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src_texts, trg):
        with torch.no_grad():
            bert_inputs = self.bert_tokenizer(
                src_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            ).to(device)
            
            bert_output = self.bert(**bert_inputs)
            src_emb = bert_output.last_hidden_state

        trg_emb = self.dec_embed(trg)
        trg_mask = self.generate_square_subsequent_mask(trg.size(1))
        
        output = self.transformer(
            src_emb, 
            self.dropout(trg_emb),
            tgt_mask=trg_mask,
            src_key_padding_mask=(bert_inputs.input_ids == self.bert_tokenizer.pad_token_id),
            tgt_key_padding_mask=(trg == 0)
        )
        
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(device)

class TRCEDataset(Dataset):
    def __init__(self, articles, highlights, vocab):
        self.articles = articles 
        self.highlights = [self.text_to_ids(s, vocab, add_special=True) for s in highlights]
        
    def text_to_ids(self, text, vocab, add_special=False):
        ids = [vocab.word2idx.get(word, 1) for word in text.split()]
        if add_special:
            ids = [vocab.word2idx["<sos>"]] + ids + [vocab.word2idx["<eos>"]]
        return ids
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        return (
            self.articles[idx],
            torch.tensor(self.highlights[idx])
        )

trce_train_dataset = TRCEDataset(train_articles, train_highlights, vocab)
trce_train_loader = DataLoader(trce_train_dataset, batch_size=2, shuffle=True)

model_trce = TRCEModel(VOCAB_SIZE).to(device)
optimizer_trce = torch.optim.Adam(model_trce.parameters(), lr=0.0001)

for epoch in range(2):  
    model_trce.train()
    total_loss = 0
    
    for batch_idx, (src_texts, trg) in enumerate(trce_train_loader):
        trg = trg.to(device)
        
        outputs = model_trce(src_texts, trg[:, :-1])
        
        loss = F.cross_entropy(
            outputs.reshape(-1, VOCAB_SIZE), 
            trg[:, 1:].reshape(-1), 
            ignore_index=0
        )
        
        optimizer_trce.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_trce.parameters(), 1.0)
        optimizer_trce.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"TRCE Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
    
    print(f"TRCE Epoch {epoch+1} Average Loss: {total_loss/len(trce_train_loader):.4f}")

print("TRCE Training completed!")

def generate_trce_summary(model, article_text, vocab, max_len=20):
    model.eval()
    with torch.no_grad():
        bert_inputs = model.bert_tokenizer(
            article_text, 
            return_tensors='pt', 
            truncation=True
        ).to(device)
        
        bert_output = model.bert(**bert_inputs)
        src_emb = bert_output.last_hidden_state

        generated = torch.tensor([[vocab.word2idx["<sos>"]]]).to(device)
        
        for _ in range(max_len):
            trg_emb = model.dec_embed(generated)
            output = model.transformer.decoder(
                trg_emb, 
                src_emb,
                memory_key_padding_mask=(bert_inputs.input_ids == model.bert_tokenizer.pad_token_id)
            )
            next_token = model.fc_out(output[:, -1, :]).argmax(-1)
            
            if next_token == vocab.word2idx["<eos>"]:
                break
                
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    summary = [vocab.idx2word.get(idx.item(), "<unk>") for idx in generated.squeeze()[1:]]
    return " ".join(summary)


# CELL 4: REINFORCEMENT-LEARNING MODEL
# In contrast to the above described model that minimizes a loss function,
# the RL model uses self-critical sequence training to maximize a specific reward.

def compute_reward(generated, reference):
    gen_tokens = set(generated.split())
    ref_tokens = set(reference.split())
    return len(gen_tokens & ref_tokens) / (len(ref_tokens) + 1e-8)

def sample_summary_rl(model, src_graph, vocab, max_len=20):
    model.eval()
    tokenized = [vocab.word2idx.get(word, 1) for word in src_graph.split()]
    src = torch.tensor([tokenized]).to(device)
    
    enc_embedded = model.enc_embed(src)
    enc_out, (h_n, c_n) = model.encoder(enc_embedded)
    h_n = torch.cat([h_n[0], h_n[1]], dim=-1)
    c_n = torch.cat([c_n[0], c_n[1]], dim=-1)
    
    decoder_hidden = model.hidden_proj(h_n).unsqueeze(0)
    decoder_cell = model.cell_proj(c_n).unsqueeze(0)
    
    decoder_input = torch.tensor([[vocab.word2idx["<sos>"]]]).to(device)
    summary = []
    log_prob_sum = 0.0
    
    for _ in range(max_len):
        dec_embedded = model.dec_embed(decoder_input)
        dec_out, (decoder_hidden, decoder_cell) = model.decoder(dec_embedded, (decoder_hidden, decoder_cell))
        output = model.fc(dec_out)
        probs = F.softmax(output[:, -1, :], dim=-1)
        dist = torch.distributions.Categorical(probs)
        next_token = dist.sample()
        log_prob = dist.log_prob(next_token)
        log_prob_sum += log_prob
        next_token_id = next_token.item()
        if next_token_id == vocab.word2idx["<eos>"]:
            break
        summary.append(vocab.idx2word.get(next_token_id, "<unk>"))
        decoder_input = torch.tensor([[next_token_id]]).to(device)
        
    return " ".join(summary), log_prob_sum

# RL training loop (using the same train_loader as AS2SP)
rl_epochs = 1
for epoch in range(rl_epochs):
    model.train()  # reusing our AS2SP model
    total_rl_loss = 0
    for batch_idx, (src, trg) in enumerate(train_loader):
        src, trg = src.to(device), trg.to(device)
        batch_size = src.size(0)
        batch_loss = 0.0
        for i in range(batch_size):
            # Reconstruct source and reference strings
            src_tokens = [vocab.idx2word.get(idx.item(), "<unk>") for idx in src[i] if idx.item() != 0]
            ref_tokens = [vocab.idx2word.get(idx.item(), "<unk>") for idx in trg[i] if idx.item() not in [0, vocab.word2idx["<sos>"], vocab.word2idx["<eos>"]]]
            src_graph = " ".join(src_tokens)
            reference = " ".join(ref_tokens)
            # Sampled summary (stochastic) and greedy summary (baseline)
            sample, log_prob_sum = sample_summary_rl(model, src_graph, vocab)
            greedy = generate_summary(model, src_graph, vocab)
            r_sample = compute_reward(sample, reference)
            r_greedy = compute_reward(greedy, reference)
            loss = - (r_sample - r_greedy) * log_prob_sum
            batch_loss += loss
        batch_loss = batch_loss / batch_size
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_rl_loss += batch_loss.item()
        if batch_idx % 10 == 0:
            print(f"RL Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {batch_loss.item():.4f}")
    print(f"RL Epoch {epoch+1} Average Loss: {total_rl_loss/len(train_loader):.4f}")


# CELL 5: PETR MODEL (PRE-TRAINED ENCODER TRANSFORMER)
from transformers import BertTokenizer, BertModel

class PETRModel(nn.Module):
    def __init__(self, vocab_size, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)  # Fine-tune BERT
        self.bert_emb_size = 768
        self.transformer = nn.Transformer(
            d_model=self.bert_emb_size,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.dec_embed = nn.Embedding(vocab_size, self.bert_emb_size)
        self.fc_out = nn.Linear(self.bert_emb_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, src_texts, trg):
        bert_inputs = self.bert_tokenizer(src_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        bert_output = self.bert(**bert_inputs)
        src_emb = bert_output.last_hidden_state
        trg_emb = self.dec_embed(trg)
        trg_mask = self.generate_square_subsequent_mask(trg.size(1))
        output = self.transformer(
            src_emb,
            self.dropout(trg_emb),
            tgt_mask=trg_mask,
            src_key_padding_mask=(bert_inputs.input_ids == self.bert_tokenizer.pad_token_id),
            tgt_key_padding_mask=(trg == 0)
        )
        return self.fc_out(output)
    
    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(device)

petr_train_dataset = TRCEDataset(train_articles, train_highlights, vocab)
petr_train_loader = DataLoader(petr_train_dataset, batch_size=2, shuffle=True)

model_petr = PETRModel(VOCAB_SIZE).to(device)
optimizer_petr = torch.optim.Adam(model_petr.parameters(), lr=0.0001)

for epoch in range(2):
    model_petr.train()
    total_loss = 0
    for batch_idx, (src_texts, trg) in enumerate(petr_train_loader):
        trg = trg.to(device)
        outputs = model_petr(src_texts, trg[:, :-1])
        loss = F.cross_entropy(outputs.reshape(-1, VOCAB_SIZE), trg[:, 1:].reshape(-1), ignore_index=0)
        optimizer_petr.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_petr.parameters(), 1.0)
        optimizer_petr.step()
        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"PETR Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
    print(f"PETR Epoch {epoch+1} Average Loss: {total_loss/len(petr_train_loader):.4f}")
print("PETR Training completed!")

def generate_petr_summary(model, article_text, vocab, max_len=20):
    model.eval()
    with torch.no_grad():
        bert_inputs = model.bert_tokenizer(article_text, return_tensors='pt', truncation=True).to(device)
        bert_output = model.bert(**bert_inputs)
        src_emb = bert_output.last_hidden_state
        generated = torch.tensor([[vocab.word2idx["<sos>"]]]).to(device)
        for _ in range(max_len):
            trg_emb = model.dec_embed(generated)
            output = model.transformer.decoder(
                trg_emb,
                src_emb,
                memory_key_padding_mask=(bert_inputs.input_ids == model.bert_tokenizer.pad_token_id)
            )
            next_token = model.fc_out(output[:, -1, :]).argmax(-1)
            if next_token == vocab.word2idx["<eos>"]:
                break
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        summary = [vocab.idx2word.get(idx.item(), "<unk>") for idx in generated.squeeze()[1:]]
    return " ".join(summary)
