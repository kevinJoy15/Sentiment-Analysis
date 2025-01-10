import numpy as np
import FileReader
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

def load_custom_dataset(train_pos_dir, train_neg_dir, dev_dir, test_dir):
    data = {
        "train": {"text": [], "label": []},
        "dev": {"text": [], "label": []},
        "test": {"text": [], "label": []},
    }

    # Load training data separately for positive and negative folders
    pos_texts, pos_labels = FileReader.load_files(train_pos_dir, label=1)
    neg_texts, neg_labels = FileReader.load_files(train_neg_dir, label=0)
    data["train"]["text"].extend(pos_texts + neg_texts)
    data["train"]["label"].extend(pos_labels + neg_labels)

    # Load development and test data
    dev_texts, dev_labels = FileReader.load_files(dev_dir, label=None)
    test_texts, test_labels = FileReader.load_files(test_dir, label=None)
    data["dev"]["text"], data["dev"]["label"] = dev_texts, dev_labels
    data["test"]["text"], data["test"]["label"] = test_texts, test_labels

    # Convert to Hugging Face Dataset format
    return DatasetDict({
        "train": Dataset.from_dict({"text": data["train"]["text"], "label": data["train"]["label"]}),
        "validation": Dataset.from_dict({"text": data["dev"]["text"], "label": data["dev"]["label"]}),
        "test": Dataset.from_dict({"text": data["test"]["text"], "label": data["test"]["label"]}),
    })

# Tokenization function
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

# Custom metric computation using sklearn
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(np(logits), axis=1)
    accuracy = accuracy_score(labels, predictions.numpy())
    return {"accuracy": accuracy}

def main():
    # Paths to your dataset folders
    train_pos_dir = "coursework/dataset/trainDataset/pos"
    train_neg_dir = "coursework/dataset/trainDataset/neg"
    dev_dir = "coursework/dataset/evaluationDataset"
    test_dir = "coursework/dataset/testDataset"

    # Load datasets
    raw_datasets = load_custom_dataset(train_pos_dir, train_neg_dir, dev_dir, test_dir)

    # Choose model name
    model_name = "bert-base-cased"  # Change to 'bert-base-uncased' to test uncased
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize datasets
    tokenized_datasets = raw_datasets.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,            
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    # Evaluate model
    eval_results = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test Results: {eval_results}")

if __name__ == "__main__":
    main()