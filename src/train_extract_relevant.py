import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    MT5ForConditionalGeneration,
    MT5TokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# 1) Paths
DATA_PATH = Path("data/extract_relevant.jsonl")
MODEL_NAME = "google/mt5-large"
OUTPUT_DIR = "models/extract-relevant-mt5"

# 2) Hyperparams
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5

def load_and_prepare():
    # Load JSONL into HuggingFace Dataset
    records = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Expect each line to have 'raw' (the text) and 'target' (the JSON output as string)
            records.append({"raw": obj["raw"], "target": obj["target"]})
    ds = Dataset.from_list(records)

    # Tokenizer & model
    tokenizer = MT5TokenizerFast.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Preprocessing function
    def preprocess_fn(batch):
        inputs = tokenizer(
            batch["raw"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                batch["target"],
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding="max_length",
            )
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized = ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    return tokenized, tokenizer, model, data_collator


def main():
    tokenized_ds, tokenizer, model, data_collator = load_and_prepare()

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="no",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,
        logging_steps=100,
        save_steps=500,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model fine-tuned and saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
