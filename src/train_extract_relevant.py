from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

def main():
    # 1. Ladda dataset
    dataset = load_dataset(
        "json",
        data_files="data/extract_relevant.jsonl",
        split="train"
    )

    # 2. Initiera tokenizer & modell
    model_name = "birgermoell/t5-base-swedish"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 3. Förbehandlingsfunktion
    def preprocess(examples):
        inputs = tokenizer(
            examples["input"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        targets = tokenizer(
            examples["output"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_ds = dataset.map(preprocess, batched=True)

    # 4. Träningsargument
    training_args = TrainingArguments(
        output_dir="models/extract-relevant",
        num_train_epochs=4,
        per_device_train_batch_size=2,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        logging_dir="models/extract-relevant/logs",
        evaluation_strategy="no"
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds
    )

    # 6. Träna och spara
    trainer.train()
    model.save_pretrained("models/extract-relevant")
    tokenizer.save_pretrained("models/extract-relevant")

if __name__ == "__main__":
    main()
