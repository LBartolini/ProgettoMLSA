from train import tokenizer, input_len
import torch
from utils import predict
from lstm import LSTM

DEVICE = 'cpu'

if __name__ == '__main__':
    checkpoint = torch.load('saves/checkpoint.pt', weights_only=False, map_location=torch.device(DEVICE))
    model = checkpoint['model']

    print(predict(model, tokenizer, "a = a +1\n if a:", input_len=input_len, verbose=True, DEVICE=DEVICE))