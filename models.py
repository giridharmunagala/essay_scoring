from torch import nn
import torch

from torch import nn
import torch

class WordSentRegressor(nn.Module):
    """
        Word Sentence Regression model.
        Args:
            word_embed_dim: Dimension of the word embeddings
            sent_embed_dim: Dimension of the sentence embeddings
    """
    def __init__(self, word_embed_dim:int, sent_embed_dim:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.word_lstm_1 = nn.LSTM(word_embed_dim, 128, batch_first=True)
        self.word_lstm_2 = nn.LSTM(128, 64, batch_first=True)
        self.sent_lstm_1 = nn.LSTM(sent_embed_dim, 128, batch_first=True)
        self.sent_lstm_2 = nn.LSTM(128, 64, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, word_embeds, sent_embeds):
        word_lstm_out, _ = self.word_lstm_1(word_embeds)
        word_lstm_out = self.relu(word_lstm_out)
        word_lstm_out, _ = self.word_lstm_2(word_lstm_out)
        word_lstm_out = self.relu(word_lstm_out)
        
        sent_lstm_out, _ = self.sent_lstm_1(sent_embeds)
        sent_lstm_out = self.relu(sent_lstm_out)
        sent_lstm_out, _ = self.sent_lstm_2(sent_lstm_out)
        sent_lstm_out = self.relu(sent_lstm_out)
        
        word_lstm_out = word_lstm_out.mean(dim=1)
        sent_lstm_out = sent_lstm_out.mean(dim=1)
        
        
        combined = torch.cat([word_lstm_out, sent_lstm_out], dim=1)
        
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.fc2(out).squeeze()
        
        return out