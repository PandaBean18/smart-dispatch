import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sklearn.model_selection import train_test_split

def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    train_examples = []
    # Data is expected to be a list of dicts: {"anchor_draft": ..., "positive_match": ..., "negative_match": ...}
    for item in data:
        train_examples.append(InputExample(texts=[item['anchor_draft'], item['positive_match'], item['negative_match']]))
    return train_examples

def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentence-transformers model for Smart Dispatch")
    parser.add_argument("--data_path", type=str, default="data/triplets.json", help="Path to synthetic dataset")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--output_dir", type=str, default="models/smart-dispatch-minilm-v1", help="Directory to save the fine-tuned model")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}")
    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found. Run generate_data.py first.")
        return
        
    dataset = load_data(args.data_path)
    
    # Split into train and validation sets (90/10)
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)
    
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    
    print(f"Loading base model: {args.model_name}")
    # We use MiniLM (~22M parameters) because it can be quantized to ~15MB for fast on-device edge execution
    model = SentenceTransformer(args.model_name)
    
    # MultipleNegativesRankingLoss is ideal for contrastive learning with triplets (anchor, positive, negative)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Prepare evaluator
    # Using TripletEvaluator on validation set
    anchors = [ex.texts[0] for ex in val_data]
    positives = [ex.texts[1] for ex in val_data]
    negatives = [ex.texts[2] for ex in val_data]
    
    evaluator = evaluation.TripletEvaluator(anchors, positives, negatives, name="smart-dispatch-val")
    
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1) # 10% of train data for warm-up
    
    print("Starting fine-tuning...")
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=100,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        show_progress_bar=True
    )
    
    print(f"Fine-tuning complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
