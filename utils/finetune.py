import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import wandb
import os
import torch

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a model using Huggingface Trainer.")
    
    # Model and dataset parameters
    parser.add_argument("--model_id", type=str, default="/shared/nas2/shared/llms/Qwen1.5-7B-Chat", help="Pretrained model ID or path.")
    parser.add_argument("--train_file", type=str, default="/shared/nas2/ph16/toxic/data/refusal/finetune_data.jsonl", help="Path to the training dataset (JSONL format).")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, default="/shared/nas2/ph16/toxic/finetuned_LM/Qwen1.5-7B-Chat", help="Output directory to save the model.")
    parser.add_argument("--num_train_epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--instruction_tuning", action="store_true")
    
    # Miscellaneous
    parser.add_argument("--logging_steps", type=int, default=20, help="Logging frequency in steps.")
    parser.add_argument("--save_strategy", type=str, default="no", choices=["no", "epoch", "steps"], help="Model saving strategy.")
    parser.add_argument("--wandb_project", type=str, default="full-finetune", help="Wandb project name.")
    parser.add_argument("--dataset", type=str, default="mmlu",)
    args = parser.parse_args()
    args.wandb_run_name = args.model_id



    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and configure model
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", torch_dtype=torch.bfloat16)
    

    # Load dataset
    dataset = load_dataset("json", data_files={"train": args.train_file})
    def preprocess(example):
        if args.instruction_tuning:
            if args.dataset == "mmlu":
                    
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": example["prompt"][0]["content"]},
                    {"role": "assistant", "content": 'The answer is {' + example["extra_info"]["answer"] + '}.'}
                ]
                return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
        else:
            assert False, "Not implemented"
            return {"text": example["text"]}
        
    dataset = dataset.map(preprocess)
    def tokenize_function(example):
        encoded = tokenizer(example["text"])
        return encoded

    tokenized_datasets = dataset.map(tokenize_function)
    train_dataset = tokenized_datasets["train"]

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy=args.save_strategy,
        logging_dir="./logs",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        report_to="wandb",
        bf16=True,
        run_name=args.wandb_run_name,
        warmup_ratio=0.2
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Save the final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    lm_head_path = os.path.join(args.output_dir, "lm_head.pth")  # Save the head separately as a .pth file (torch tensor)
    torch.save(model.lm_head.weight, lm_head_path)


if __name__ == "__main__":
    main()
