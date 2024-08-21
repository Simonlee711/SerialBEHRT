import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from tqdm import tqdm
import argparse
import logging
import os
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['WANDB_API_KEY'] = '0d3cef273ac07263f8b9035513b8693a26308dce'

# Define a custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def load_data(file_path, text_column):
    logger.info(f"Loading data from {file_path}")
    with open(file_path) as f:
        json_content = json.load(f)
    df = pd.DataFrame(json_content).T
    df['eddischarge'] = [1 if 'admitted' in s.lower() else 0 for s in df['eddischarge']] # admitted = 0, Home = 1
    df['medrecon'] = df['medrecon'].fillna("The patient was previously not taking any medications.")
    df['pyxis'] = df['pyxis'].fillna("The patient did not receive any medications.")
    df['vitals'] = df['vitals'].fillna("The patient had no vitals recorded")
    df['codes'] = df['codes'].fillna("The patient received no diagnostic codes")
    df = df.drop("admission",axis=1)
    df = df.drop("discharge",axis=1)
    df = df.drop("eddischarge_category",axis=1)
    df["info"] = df['arrival'] + " " + df["codes"] + " "+df["triage"] + " "+ df["vitals"] +" "+ df["pyxis"] +" "+ df["medrecon"]
    df = df.drop("arrival", axis=1)
    df = df.drop("codes", axis=1)
    df = df.drop("triage", axis=1)
    df = df.drop("vitals", axis=1)
    df = df.drop("pyxis", axis=1)
    df = df.drop("medrecon", axis=1)
    df = df[[col for col in df.columns if col != 'eddischarge'] + ['eddischarge']] # rearrange column to the end
    logger.info(f"Data loaded. Shape: {df.shape}")
    return df[text_column].tolist()

def train(args):
    # Initialize wandb
    wandb.init(project="PseudonotesBERT", config=vars(args))

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    # Load your data
    texts = load_data(args.data_file, args.text_column)

    # Split the data into train and validation sets
    train_texts, val_texts = train_test_split(texts, test_size=args.val_split, random_state=42)

    # Create datasets and dataloaders
    train_dataset = TextDataset(train_texts, tokenizer, max_length=args.max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Set up the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model.to(device)

    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = args.early_stopping_patience

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", leave=False)
        
        for batch in train_pbar:
            global_step += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Calculate gradient norm
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            # Log metrics to wandb
            wandb.log({
                "step": global_step,
                "train_loss": loss.item(),
                "train_perplexity": perplexity.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "total_grad_norm": total_grad_norm,
            })

            # Update progress bar
            train_pbar.set_postfix({'train_loss': f'{loss.item():.4f}', 'train_perplexity': f'{perplexity.item():.4f}'})

            # Perform validation every N steps
            if global_step % args.validation_steps == 0:
                model.eval()
                total_val_loss = 0
                total_val_perplexity = 0
                val_pbar = tqdm(val_loader, desc=f"Step {global_step} - Validation", leave=False)
                
                with torch.no_grad():
                    for val_batch in val_pbar:
                        val_input_ids = val_batch['input_ids'].to(device)
                        val_attention_mask = val_batch['attention_mask'].to(device)
                        
                        val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_input_ids)
                        val_loss = val_outputs.loss
                        val_perplexity = torch.exp(val_loss)
                        
                        total_val_loss += val_loss.item()
                        total_val_perplexity += val_perplexity.item()

                        val_pbar.set_postfix({'val_loss': f'{val_loss.item():.4f}', 'val_perplexity': f'{val_perplexity.item():.4f}'})

                avg_val_loss = total_val_loss / len(val_loader)
                avg_val_perplexity = total_val_perplexity / len(val_loader)
                
                wandb.log({
                    "step": global_step,
                    "val_loss": avg_val_loss,
                    "val_perplexity": avg_val_perplexity,
                })

                logger.info(f"Step {global_step} - Validation loss: {avg_val_loss:.4f}, Validation perplexity: {avg_val_perplexity:.4f}")

                # Save the best model and check for early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_save_path = os.path.join(args.output_dir, f'best_model_step_{global_step}.pt')
                    torch.save(model.state_dict(), model_save_path)
                    logger.info(f"Best model saved at step {global_step}: {model_save_path}")
                    
                    # Log best model as artifact
                    artifact = wandb.Artifact(f"best_model_step_{global_step}", type="model")
                    artifact.add_file(model_save_path)
                    wandb.log_artifact(artifact)
                    
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    logger.info(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {global_step} steps")
                    return

                model.train()

    logger.info("Training completed!")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue pre-training SciBERT model")
    parser.add_argument("--model_name", type=str, default="sciBERT/", help="Name or path of the pre-trained model")
    parser.add_argument("--data_file", type=str, default="/opt/data/commonfilesharePHI/jnchiang/projects/er-pseudonotes/mimic/mimic-iv-ed-2.2/mimic-iv-ed-2.2/text_repr.json", help="Path to the JSON file containing the training data")
    parser.add_argument("--text_column", type=str, required=True, help="Name of the column in the JSON file containing the text data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the trained models")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation set split ratio")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of validation rounds with no improvement after which training will be stopped")
    parser.add_argument("--validation_steps", type=int, default=10000, help="Number of steps between validation rounds")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up file logging
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    train(args)
