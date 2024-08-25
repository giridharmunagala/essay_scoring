
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import compute_quadratic_weighted_kappa

class TrainerForEssayScoring:
    """
    Trainer class for training the WordSentRegressor model.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, has_stats_features:bool=False, optimizer: optim.Optimizer = None, criterion: nn.Module = None,
                 device: str = torch.device('cuda')):
        """
        Initializes the TrainerForEssayScoring class.

        Args:
            model (nn.Module): The model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            optimizer (optim.Optimizer, optional): Optimizer for training the model. Defaults to Adam.
            criterion (nn.Module, optional): Loss function. Defaults to MSELoss.
            device (str, optional): Device to run the training on. Defaults to 'cuda'.
        """
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer if optimizer else optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = criterion if criterion else nn.MSELoss()
        self.has_stats_features = has_stats_features
        
    def train(self, num_epochs):
        """
            Trains the model for the given number of epochs.
            Requires the data  loader to have word_embeddings, sentence_embeddings & labels keys in the dictionary.
        """
        best_loss = None
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                if self.has_stats_features:
                    stats_features = batch['stats_features'].to(self.device)
                word_embedded, sent_embedded, targets = batch['word_embeddings'].to(self.device), batch['sentence_embeddings'].to(self.device), batch['labels'].to(self.device)
                self.optimizer.zero_grad()
                if self.has_stats_features:
                    outputs = self.model(word_embedded, sent_embedded, stats_features)
                else:
                    outputs = self.model(word_embedded, sent_embedded)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * word_embedded.size(0)
                # Print loss every 100 batches
                if i % 100 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
            
            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
            
            # If validation is better than previous replaces the current model with the best model
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    if self.has_stats_features:
                        stats_features = batch['stats_features'].to(self.device)
                    word_embedded, sent_embedded, targets = batch['word_embeddings'].to(self.device), batch['sentence_embeddings'].to(self.device), batch['labels'].to(self.device)
                    if self.has_stats_features:
                        outputs = self.model(word_embedded, sent_embedded, stats_features)
                    else:
                        outputs = self.model(word_embedded, sent_embedded)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * word_embedded.size(0)
            
            val_loss /= len(self.val_loader.dataset)
            if epoch == 0:
                best_loss = epoch_loss
            else:
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_model.pth')
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

class Evaluator:
    def __init__(self, model: nn.Module, test_loader: DataLoader, device: str = torch.device('cuda')):
        """
        Initializes the Evaluator class.

        Args:
            model (nn.Module): The model to be evaluated.
            test_loader (DataLoader): DataLoader for the test dataset.
            device (str, optional): Device to run the evaluation on. Defaults to 'cuda'.
        """
        self.device = device
        self.model = model.to(device)
        self.test_loader = test_loader

    def evaluate(self,evaluation_metric = 'qwk'):
        """
            Evaluates the model on the test dataset.
            Requires the data loader to have word_embeddings, sentence_embeddings & labels keys in the dictionary.
            Evaluation metric can be mse or rmse or qwk, defaults to qwk.
        """
        self.model.eval()
        test_predictions = list()
        labels = list()

        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                word_embedded, sent_embedded, targets = batch['word_embeddings'].to(self.device), batch['sentence_embeddings'].to(self.device), batch['labels'].to(self.device)
                outputs = self.model(word_embedded, sent_embedded)
                # Rounding predictions to the nearest integer
                outputs = torch.round(outputs)
                test_predictions.extend(outputs.cpu().numpy())
                labels.extend(targets.cpu().numpy())
        
        qwk = compute_quadratic_weighted_kappa(labels, test_predictions)
                
        
        print(f'Quadratic weighted kappa is : {qwk:.4f}')
        return qwk