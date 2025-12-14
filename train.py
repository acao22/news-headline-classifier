"""
Training script for News Headline Classifier.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import os
import argparse
from collections import Counter

from preprocess import prepare_data
from model import NewsClassifier, TextTokenizer


class HeadlineDataset(Dataset):
    """pytorch dataset for headlines."""
    
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.encode([text])[0]  # remove batch dimension
        return tokens, torch.tensor(label, dtype=torch.long)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_texts, batch_labels in dataloader:
        batch_texts = batch_texts.to(device)
        batch_labels = batch_labels.to(device)
        
        # forward pass
        optimizer.zero_grad()
        logits = model(batch_texts)
        loss = criterion(logits, batch_labels)
        
        # backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_texts, batch_labels in dataloader:
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(batch_texts)
            loss = criterion(logits, batch_labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='train news headline classifier')
    parser.add_argument('--csv_path', type=str, default='url_only_data.csv',
                        help='Path to CSV file')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate (reduced for more stable training)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--max_length', type=int, default=50,
                        help='Maximum sequence length')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='Directory to save model and tokenizer')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='early stopping patience (increased to allow more training)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate (increased to reduce overfitting)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for L2 regularization')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights to handle imbalance')
    
    args = parser.parse_args()
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # load and prepare data
    print("Loading data...")
    X, y = prepare_data(args.csv_path)
    print(f"Loaded {len(X)} samples")
    
    # split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # calculate class weights for imbalanced data
    class_counts = Counter(y_train)
    total = len(y_train)
    if args.use_class_weights:
        class_weights = torch.tensor([
            total / (2 * class_counts[0]),  # weight for class 0 (fox)
            total / (2 * class_counts[1])   # weight for class 1 (nbc)
        ], dtype=torch.float32).to(device)
        print(f"Class distribution - Fox: {class_counts[0]}, NBC: {class_counts[1]}")
        print(f"Class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = None
        print(f"Class distribution - Fox: {class_counts[0]}, NBC: {class_counts[1]}")
        print("Class weights: None (balanced)")
    
    # build tokenizer
    print("Building vocabulary...")
    tokenizer = TextTokenizer(vocab_size=args.vocab_size, max_length=args.max_length)
    tokenizer.build_vocab(X_train)
    print(f"Vocabulary size: {len(tokenizer.word_to_idx)}")
    
    # create datasets
    train_dataset = HeadlineDataset(X_train, y_train, tokenizer)
    val_dataset = HeadlineDataset(X_val, y_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # create model with increased dropout
    model = NewsClassifier(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        num_classes=2,
        dropout=args.dropout,
        max_length=args.max_length
    ).to(device)
    
    # attach tokenizer to model for predict() method
    model.tokenizer = tokenizer
    
    # loss with class weights to handle imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    
    # optimizer with weight decay (l2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    
    # training loop - track both accuracy and f1 score
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        # calculate f1 score for better imbalanced data evaluation
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # save best model based on f1 score (better for imbalanced data)
        improved = False
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            improved = True
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not improved:
                improved = True
        
        if improved:
            patience_counter = 0
            
            # save model with tokenizer embedded in the checkpoint
            # this way we only need model.pt, not a separate tokenizer.pkl file
            # save in format compatible with evaluator: {'state_dict': ..., 'tokenizer': ...}
            model_path = os.path.join(args.save_dir, 'model.pt')
            state_dict = model.state_dict()
            checkpoint = {
                'state_dict': state_dict,  # evaluator looks for this key
                'tokenizer': {
                    'word_to_idx': tokenizer.word_to_idx,
                    'idx_to_word': tokenizer.idx_to_word,
                    'vocab_size': tokenizer.vocab_size,
                    'max_length': tokenizer.max_length
                }
            }
            torch.save(checkpoint, model_path)
            
            # also save tokenizer separately for local use (optional)
            tokenizer_path = os.path.join(args.save_dir, 'tokenizer.pkl')
            tokenizer.save(tokenizer_path)
            
            print(f"  ✓ Saved best model (val_acc: {val_acc:.4f}, val_f1: {val_f1:.4f})")
        else:
            patience_counter += 1
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  No improvement (patience: {patience_counter}/{args.early_stopping}, lr: {current_lr:.6f})")
            if patience_counter >= args.early_stopping:
                print(f"  Early stopping after {epoch+1} epochs")
                print(f"  Best val_acc: {best_val_acc:.4f}, Best val_f1: {best_val_f1:.4f}")
                break
    
    # final evaluation
    print("\n" + "="*50)
    print("Final Results:")
    print("="*50)
    print(classification_report(val_labels, val_preds))
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation F1 Score: {best_val_f1:.4f}")
    
    # load best model for final test
    checkpoint = torch.load(os.path.join(args.save_dir, 'model.pt'), map_location='cpu')
    # check if tokenizer is embedded (new format) or just state dict (old format)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        if 'tokenizer' in checkpoint:
            model._tokenizer_data = checkpoint['tokenizer']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'tokenizer' in checkpoint:
            model._tokenizer_data = checkpoint['tokenizer']
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"\nModel saved to: {os.path.join(args.save_dir, 'model.pt')} (with embedded tokenizer)")
    print(f"Tokenizer also saved separately to: {os.path.join(args.save_dir, 'tokenizer.pkl')} (for local use)")


if __name__ == '__main__':
    main()

