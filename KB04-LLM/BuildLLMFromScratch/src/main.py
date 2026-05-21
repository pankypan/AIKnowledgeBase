from datetime import datetime
from pathlib import Path

import torch
import tiktoken

from src.model_structure.gpt_model import GPTModel
from src.config import GPT_CONFIG_124M, PROJECT_ROOT
from src.utils import token_ids_to_text, text_to_token_ids, calc_loss_loader, get_device
from src.data_process import download_and_load_verdict_text, create_dataloaders
from src.train import train_model_simple
from src.inference import generate_text_simple, generate
from src.load_pretrained_weights import load_weights_into_gpt
from src.plot_views import plot_gelu_relu_activation_functions, plot_losses


PART_MAP = {
    "1": "Initialize model and generate text",
    "2": "Train GPT with unlabeled data",
    "3": "Load pretrained weights and inference",
}


def poc01_init_model_and_generate_text():
    # plot GELU and ReLU activation functions
    import torch.nn as nn
    from src.model_structure.common_layers import GELU

    plot_gelu_relu_activation_functions(GELU(), nn.ReLU())

    # initialize model and generate text
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval() # disable dropout

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=GPT_CONFIG_124M["context_length"]
    )

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


def _cal_initial_loss(train_loader, val_loader):
    device = get_device()
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval();  # Disable dropout during inference
    model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


def train_and_save_model(base_line_prompt: str) -> Path:
    text_data = download_and_load_verdict_text()
    train_loader, val_loader = create_dataloaders(text_data, train_ratio=0.90)
    
    # calculate initial loss
    _cal_initial_loss(train_loader, val_loader)

    # train model
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()
    print(f"Using {device} device.")
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 20
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=base_line_prompt, tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    weights_path = PROJECT_ROOT / "models" / f"gpt_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    if not weights_path.parent.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), weights_path)
    return weights_path


def inference_with_trained_model(prompt: str, weights_path):
    device = get_device()

    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    return token_ids_to_text(token_ids, tokenizer)


def poc02_train_gpt_with_unlabeled_data():
    base_line_prompt = "Every effort moves you"

    weights_path = train_and_save_model(base_line_prompt)

    print("\n\nInference with trained model:")
    result = inference_with_trained_model(base_line_prompt, weights_path)
    print("Output text:\n", result)


def poc03_load_pretrained_weights_and_inference():
    from gpt_download import download_and_load_gpt2

    prompt = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must"

    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # Copy the base configuration and update with specific model settings
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    load_weights_into_gpt(gpt, params)
    device = get_device()
    gpt.to(device)

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )
    return token_ids_to_text(token_ids, tokenizer)


def print_tip(part_number: str):
    import shutil
    term_width = shutil.get_terminal_size().columns
    if part_number not in PART_MAP:
        msg = f"  ✗ Invalid part: '{part_number}'.  Valid options: {', '.join(PART_MAP.keys())}  "
        print(f"\n{msg.center(term_width)}\n")
        return
    title = f"  Part {part_number}  ·  {PART_MAP[part_number]}  "
    inner_width = term_width - 2
    title_padded = title.center(inner_width)
    print(f"\n╔{'═' * inner_width}╗")
    print(f"║{title_padded}║")
    print(f"╚{'═' * inner_width}╝\n")


if __name__ == "__main__":
    run_part = input("Enter the part number to run: ").strip()
    print_tip(run_part)

    if run_part == "1":
        poc01_init_model_and_generate_text()
    elif run_part == "2":
        poc02_train_gpt_with_unlabeled_data()
    elif run_part == "3":
        poc03_load_pretrained_weights_and_inference()
    else:
        print("Invalid part number")
