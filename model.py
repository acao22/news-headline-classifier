"""
Model module for News Headline Classifier.
Provides NewsClassifier class that can be instantiated without arguments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from collections import Counter
import numpy as np


class NewsClassifier(nn.Module):
    """
    LSTM-based classifier for news headlines.
    Uses word embeddings and bidirectional LSTM for classification.
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=128, 
                 num_layers=2, num_classes=2, dropout=0.3, max_length=50, weights_path=None):
        super(NewsClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_length = max_length
        
        # embedding layer - converts word indices to dense vectors
        # each word gets a 128-dimensional vector that captures its meaning
        # we learn these embeddings during training so similar words get similar vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # FIXING: added dropout here to prevent overfitting - randomly zeros out some embeddings
        # using lighter dropout (50% of main rate) since embeddings are early in the network
        self.embedding_dropout = nn.Dropout(dropout * 0.5)
        
        # bidirectional lstm - main part of our model!
        # reads the headline both left-to-right and right-to-left to capture context
        # 2 layers stacked on top of each other for deeper understanding
        # hidden_dim=128 means each direction has 128 units, so output is 256 total
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # attention mechanism - helps the model focus on important words
        # instead of just using the last lstm output, we weight all positions
        # so like words like "trump" or "biden" might get higher attention weights
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # added dropout here too, but even lighter (30% of main rate) since attention is sensitive
        self.attention_dropout = nn.Dropout(dropout * 0.3)
        
        # classifier head - takes the lstm output and makes the final prediction
        # originally had 2 layers, but we added a third layer to give more capacity
        # with proper regularization (dropout) this helps without overfitting
        # fc1: 256 -> 128 (lstm output to first hidden layer)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)  # main dropout rate
        # fc2: 128 -> 64 (added this extra layer for more capacity)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout * 0.8)  # slightly less dropout here
        # fc3: 64 -> 2 (final layer outputs logits for fox vs nbc)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        # note: weights_path is accepted for evaluator compatibility but weights
        # are loaded by the evaluator itself via load_state_dict()
        
        # these will be set when model.pt is loaded (tokenizer embedded in checkpoint)
        self._tokenizer_data = None
        self.tokenizer = None  # will be set when tokenizer is loaded
    
    def _load_tokenizer_from_state(self):
        """extract tokenizer from model state if it was embedded"""
        # first check if we already loaded it
        if self.tokenizer is not None:
            return True
            
        if hasattr(self, '_tokenizer_data') and self._tokenizer_data is not None:
            tokenizer = TextTokenizer()
            tokenizer.word_to_idx = self._tokenizer_data['word_to_idx']
            tokenizer.idx_to_word = self._tokenizer_data['idx_to_word']
            tokenizer.vocab_size = self._tokenizer_data['vocab_size']
            tokenizer.max_length = self._tokenizer_data['max_length']
            tokenizer.vocab_built = True
            self.tokenizer = tokenizer
            return True
        
        # try to load from model.pt if it exists (tokenizer embedded in checkpoint)
        # try multiple possible paths
        possible_paths = ['model.pt', './model.pt', os.path.join(os.path.dirname(__file__), 'model.pt')]
        for checkpoint_path in possible_paths:
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'tokenizer' in checkpoint:
                        self._tokenizer_data = checkpoint['tokenizer']
                        tokenizer = TextTokenizer()
                        tokenizer.word_to_idx = self._tokenizer_data['word_to_idx']
                        tokenizer.idx_to_word = self._tokenizer_data['idx_to_word']
                        tokenizer.vocab_size = self._tokenizer_data['vocab_size']
                        tokenizer.max_length = self._tokenizer_data['max_length']
                        tokenizer.vocab_built = True
                        self.tokenizer = tokenizer
                        return True
                except Exception as e:
                    continue  # try next path
        
        return False
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to also try to extract tokenizer from checkpoint.
        This helps when the evaluator loads the checkpoint.
        """
        # try to extract tokenizer if we're loading from a full checkpoint
        # (though evaluator usually only passes state_dict)
        result = super().load_state_dict(state_dict, strict=strict)
        
        # after loading, try to load tokenizer from model.pt
        # this ensures tokenizer is available even if evaluator only passed state_dict
        self._load_tokenizer_from_state()
        
        return result
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, seq_length) with token indices,
               or list of strings (will be tokenized automatically)
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # handle string inputs - the backend evaluator passes raw strings
        # so we need to tokenize them first if they're not already numbers
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], str):
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                # try to load tokenizer from model state first (embedded in model.pt)
                if not self._load_tokenizer_from_state():
                    # fallback: try to load from file
                    tokenizer_path = 'tokenizer.pkl'
                    if os.path.exists(tokenizer_path):
                        tokenizer = TextTokenizer()
                        tokenizer.load(tokenizer_path)
                        self.tokenizer = tokenizer
                    else:
                        raise ValueError(
                            f"Model needs tokenizer to process string inputs. "
                            f"Tokenizer should be embedded in model.pt or at '{tokenizer_path}'"
                        )
            device = next(self.parameters()).device
            x = self.tokenizer.encode(x).to(device)
        
        # step 1: convert word indices to embeddings
        # each word becomes a 128-dim vector that we learned during training
        embedded = self.embedding(x)
        # apply dropout to embeddings - randomly zero out some to prevent overfitting
        embedded = self.embedding_dropout(embedded)
        
        # step 2: pass through bidirectional lstm
        # this reads the headline in both directions and captures context
        # output has shape (batch, sequence_length, 256) - 128 from each direction
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # step 3: attention pooling - figure out which words are important
        # apply light dropout to attention input
        attention_input = self.attention_dropout(lstm_out)
        # compute attention weights for each word position
        attention_weights = self.attention(attention_input)
        # normalize weights so they sum to 1 (softmax)
        attention_weights = F.softmax(attention_weights, dim=1)
        # weighted sum - words with higher attention contribute more
        # this gives us a single 256-dim vector representing the whole headline
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # step 4: classification - make the final prediction
        # pass through FC layers w/ dropout
        out = self.fc1(attended)
        out = F.relu(out)  # relu activation adds non-linearity
        out = self.dropout1(out)  # dropout prevents overfitting
        
        # second layer (FIXING: adde this to give more capacity)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout2(out)  # slightly less dropout here
        
        # final layer outputs 2 logits - one for fox, one for nbc
        # higher logit = more confident prediction
        logits = self.fc3(out)
        
        return logits
    
    def predict(self, batch):
        """
        Predict method for backend compatibility.
        The backend evaluator calls this method with batches of headlines.
        
        Args:
            batch: List of headline strings, or tensor of tokenized sequences
            
        Returns:
            predictions: List/array of predicted class indices (0 or 1)
        """
        # set model to evaluation mode - turns off dropout and batch norm updates
        self.eval()
        device = next(self.parameters()).device
        
        # no gradient computation during inference (faster and uses less memory)
        with torch.no_grad():
            # handle string inputs - backend passes raw headline strings
            if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], str):
                # need to tokenize strings first
                if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                    # try to load tokenizer from model state first (embedded in model.pt)
                    if not self._load_tokenizer_from_state():
                        # fallback: try to load from file
                        tokenizer_path = 'tokenizer.pkl'
                        if os.path.exists(tokenizer_path):
                            tokenizer = TextTokenizer()
                            tokenizer.load(tokenizer_path)
                            self.tokenizer = tokenizer
                        else:
                            raise ValueError(
                                f"Model needs tokenizer to process string inputs. "
                                f"Tokenizer should be embedded in model.pt or at '{tokenizer_path}'"
                            )
                batch_tensor = self.tokenizer.encode(batch)
                batch_tensor = batch_tensor.to(device)
            elif isinstance(batch, torch.Tensor):
                # already tokenized, just move to right device
                batch_tensor = batch.to(device)
            else:
                # try to convert to tensor if it's some other format
                batch_tensor = torch.tensor(batch, dtype=torch.long).to(device)
            
            # run forward pass to get logits
            logits = self(batch_tensor)
            
            # convert logits to predictions - take the class with highest logit
            if isinstance(logits, torch.Tensor):
                predictions = torch.argmax(logits, dim=-1).cpu()
                # return as list for backend compatibility
                if len(predictions.shape) == 0:  # single prediction
                    return predictions.item()
                return predictions.numpy().tolist()
            else:
                # fallback for numpy arrays
                return np.argmax(logits, axis=-1).tolist()


class TextTokenizer:
    """
    Simple tokenizer that builds vocabulary from training data.
    Converts headlines (strings) into sequences of numbers that the model can understand.
    """
    
    def __init__(self, vocab_size=10000, max_length=50):
        self.vocab_size = vocab_size
        self.max_length = max_length
        # special tokens: pad for short sequences, unk for unknown words
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_built = False
        
    def build_vocab(self, texts):
        """Build vocabulary from list of texts."""
        # count how many times each word appears in  training data
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # get the most common words - we only keep top 10k to limit vocabulary size
        # -2 because we already have PAD and UNK tokens
        most_common = word_counts.most_common(self.vocab_size - 2)
        
        # assign each word a unique number
        for word, count in most_common:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab_built = True
        
    def encode(self, texts, max_length=None):
        """
        Encode texts to token indices.
        Takes a list of headline strings and converts them to sequences of numbers.
        
        Args:
            texts: List of strings
            max_length: Maximum sequence length (defaults to self.max_length)
            
        Returns:
            Tensor of shape (batch_size, seq_length)
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        max_len = max_length or self.max_length
        encoded = []
        
        for text in texts:
            # split into words & convert to lowercase
            words = text.lower().split()
            # convert each word to its index, use UNK (1) if word not in vocabulary
            indices = [self.word_to_idx.get(word, 1) for word in words[:max_len]]
            # pad or truncate to make all sequences same length
            if len(indices) < max_len:
                # pad with zeros (PAD token) if sequence is too short
                indices.extend([0] * (max_len - len(indices)))
            else:
                # truncate if sequence is too long
                indices = indices[:max_len]
            encoded.append(indices)
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def save(self, path):
        """Save tokenizer to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'vocab_built': self.vocab_built
            }, f)
    
    def load(self, path):
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = data['idx_to_word']
            self.vocab_size = data['vocab_size']
            self.max_length = data['max_length']
            self.vocab_built = data['vocab_built']


def get_model():
    """
    Factory function to create model instance.
    Alternative entry point for backend - some backends prefer this over direct class instantiation.
    """
    return NewsClassifier()
