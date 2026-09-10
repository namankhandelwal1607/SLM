from fastapi import FastAPI
from pydantic import BaseModel

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

import torch

from model import IPLGPT, IPLGPTConfig


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "./best_model.pt"
VOCAB_PATH = "./vocab.json"
MERGES_PATH = "./merges.txt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="IPL GPT API",
    description="API for my custom Small Language Model",
    version="1.0"
)


# ============================================================
# Load Tokenizer
# ============================================================

tokenizer = Tokenizer(
    BPE(
        vocab=VOCAB_PATH,
        merges=MERGES_PATH
    )
)

tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

print("Tokenizer loaded successfully.")

vocab_size = tokenizer.get_vocab_size()

print("Vocabulary size:", vocab_size)


# ============================================================
# Load Checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

print("Checkpoint loaded successfully.")

print("Checkpoint keys:", checkpoint.keys())


# ============================================================
# Load Exact Model Configuration
# ============================================================

model_config = checkpoint["config"]

print("Model configuration:")
print(model_config)


config = IPLGPTConfig(
    vocab_size=model_config["vocab_size"],
    block_size=model_config["block_size"],
    n_layer=model_config["n_layer"],
    n_head=model_config["n_head"],
    n_embd=model_config["n_embd"]
)


# ============================================================
# Create Model
# ============================================================

model = IPLGPT(config)


# ============================================================
# Load Trained Weights
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# Move Model to Device
# ============================================================

model.to(DEVICE)

model.eval()


print(f"Model loaded successfully on: {DEVICE}")


# ============================================================
# Request Schema
# ============================================================

class GenerateRequest(BaseModel):

    prompt: str

    max_new_tokens: int = 100

    temperature: float = 1.0

    top_k: int = 50


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "IPL GPT API is running"
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "vocab_size": vocab_size
    }


# ============================================================
# Generate Endpoint
# ============================================================

@app.post("/generate")
def generate(request: GenerateRequest):

    prompt = request.prompt

    max_new_tokens = request.max_new_tokens

    temperature = request.temperature

    top_k = request.top_k


    # ========================================================
    # Encode Prompt
    # ========================================================

    encoded = tokenizer.encode(prompt)

    input_ids = encoded.ids


    idx = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )


    # ========================================================
    # Generate
    # ========================================================

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep context within model's maximum size
            idx_cond = idx[
                :, -config.block_size:
            ]


            # Forward pass
            logits = model(idx_cond)


            # Last token's logits
            logits = logits[:, -1, :]


            # Temperature
            logits = logits / temperature


            # =================================================
            # Top-k Sampling
            # =================================================

            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                logits[
                    logits < values[:, [-1]]
                ] = float("-inf")


            # =================================================
            # Probabilities
            # =================================================

            probabilities = torch.softmax(
                logits,
                dim=-1
            )


            # =================================================
            # Sample Next Token
            # =================================================

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )


            # Append token
            idx = torch.cat(
                (idx, next_token),
                dim=1
            )


    # ========================================================
    # Decode
    # ========================================================

    generated_ids = idx[0].tolist()

    output = tokenizer.decode(
        generated_ids
    )


    return {
        "prompt": prompt,
        "response": output
    }