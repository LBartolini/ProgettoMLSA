from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from lstm import LSTM
from transformer import TransformerModel

input_len = 30  # Number of tokens in input
target_len = 1 # Number of tokens to predict
batch_size = 512 # Batch size

FROM_SCRATCH = True
DEVICE = 'cpu'

model_type = 'LSTM'
#model = 'Transformer'

tokenizer = AutoTokenizer.from_pretrained("codesage/codesage-base", trust_remote_code=True)

def create_dataset(dataset, input_len, target_len=1):
    X = []
    y = []

    print("\nCreating dataset...")

    for j, code in enumerate(dataset):
        progress_bar(j, len(dataset))
        tmp_X = []
        tmp_y = []
        err = False
        for i in range(len(code['code_tokens']) - input_len - target_len):
            try:
                tokens = [tokenizer.encode(code['code_tokens'][i+k])[0] for k in range(input_len)]
                tmp_X.append(torch.Tensor(tokens))
            except Exception as e:
                err = True
                break

        if err:
            continue

        for i in range(input_len, len(code['code_tokens']) - target_len):
            try:
                tokens = [tokenizer.encode(code['code_tokens'][i+k])[0] for k in range(target_len)]
                tmp_y.append(torch.Tensor(tokens))
            except Exception as e:
                err = True
                break

        if not err:
            X.extend(tmp_X)
            y.extend(tmp_y)

    return torch.stack(X), torch.stack(y)


if __name__ == '__main__':
    if FROM_SCRATCH:
        dataset = load_dataset("Nan-Do/code-search-net-python", trust_remote_code=True)
        X, y = create_dataset(dataset['train'].select(range(25000, 25100)), input_len)
        torch.save({'X': X, 'y': y}, "saves/dataset.pt")

        pretrained_emb = None
        dict_size = int(X.max())+1
        emb_dim = 128

        if model_type=='LSTM':
            model = LSTM(dictionary_size=dict_size,
                    embedding_dim=emb_dim,
                    batch_size=batch_size,
                    pretrained_emb=pretrained_emb,
                    DEVICE=DEVICE)
            
        elif model_type=='Transformer':
            transformer = nn.Transformer(d_model=emb_dim, 
                                         nhead=4, 
                                         num_encoder_layers=3, 
                                         num_decoder_layers=3, 
                                         batch_first=True).to(DEVICE)
            model = TransformerModel(transformer, 
                                     input_len=input_len, 
                                     target_len=target_len, 
                                     dict_size=dict_size, 
                                     embedding_dim=emb_dim, 
                                     pretrained_emb=pretrained_emb, 
                                     DEVICE=DEVICE)
            
        optimizer = optim.Adam(model.parameters(), lr=0.002)

    else:
        emb_model = AutoModel.from_pretrained("codesage/codesage-base", trust_remote_code=True)
        dataset = torch.load("saves/dataset.pt")
        X, y = dataset['X'], dataset['y']
        
        pretrained_emb = emb_model.get_input_embeddings()
        dict_size = pretrained_emb.num_embeddings
        emb_dim = pretrained_emb.embedding_dim

        checkpoint = torch.load('saves/checkpoint.pt', weights_only=False, map_location=torch.device(DEVICE))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        
    usable_size= (X.shape[0] // (batch_size)) * batch_size
    X_trunc, y_trunc = X[:usable_size, :], y[:usable_size, :]

    full_dataset = TensorDataset(X_trunc.to(torch.long), y_trunc.to(torch.float))
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    model, _ = train_model(model, criterion, optimizer, dict_size, train_loader, valid_loader, num_epochs=1, DEVICE=DEVICE)

    state = {
        'model': model,
        'optimizer': optimizer,
        'input_len': input_len
    }
    torch.save(state, "saves/checkpoint.pt")
        
