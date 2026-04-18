# Pocket Agent Tool-Calling Fine-Tuning

This project focuses on fine-tuning a small language model (`SmolLM2-1.7B`) to function as an on-device Pocket Agent capable of strict JSON tool-calling and intelligent natural language conversation.

## 🚀 Overview

The goal of this project is to train an agent that can reliably parse user intent and map it to specific tool schemas. If a tool applies, the model outputs a valid JSON tool call wrapped in `<tool_call>` tags. If no tool applies (e.g., chitchat, impossible requests), the model politely refuses or answers in plain text.

## 🛠️ Supported Tools

The model is trained to use the following 5 tools:
- `weather`: Retrieve weather information based on location and unit (C/F).
- `calendar`: Manage calendar events (list or create).
- `convert`: Perform unit conversions.
- `currency`: Convert between currencies using ISO3 codes.
- `sql`: Execute basic SQL queries for data retrieval.

Tool schemas are strictly defined in `starter/tool_schemas.json`.

## 🗄️ Dataset Generation Pipeline (`generate_data.py`)

To create a highly diverse and robust dataset, we use a synthetic data generation pipeline powered by the **NVIDIA NIM API** (`moonshotai/kimi-k2-instruct`). 

The pipeline generates:
1. **Standard Examples:** Highly diverse requests for each tool, incorporating typos, slang, and various phrasing styles.
2. **Refusals:** Examples where no tool applies to teach the model to avoid hallucinating tool calls.
3. **Multi-Turn Context:** 2-turn conversations to train the model on context resolution (e.g., resolving pronouns in follow-up questions).

The generated data is safely appended, deduplicated via SHA-256 hashing to ensure uniqueness, and formatted into OpenAI chat templates (`data/train.jsonl`).

## 🧠 Fine-Tuning (`train.ipynb`)

We use **QLoRA** (Quantized Low-Rank Adaptation) to efficiently fine-tune the `HuggingFaceTB/SmolLM2-1.7B` base model on consumer hardware (e.g., 6 GB VRAM GPUs).

**Training Configuration:**
- **Quantization:** 4-bit `nf4` precision using `BitsAndBytes`.
- **LoRA Adapters:** Rank `r=16`, `alpha=32`, aggressively targeting attention and MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Optimization:** 3 Epochs, effective batch size of 16, `paged_adamw_8bit` optimizer, and cosine learning rate scheduling.
- **Data Formatting:** Dynamic padding via DataCollator to avoid padding token limits.

## 🗜️ Post-Training & Quantization (GGUF)

Once the model is fine-tuned, it must be optimized to run natively on mobile devices. The `train.ipynb` notebook handles the entire post-processing pipeline:

1. **Merging:** The LoRA adapter weights are securely merged into the FP16 base model weights to create a standalone model.
2. **GGUF Conversion:** Using `llama.cpp` (`convert_hf_to_gguf.py`), the HuggingFace model is converted into the optimized `.gguf` FP16 format.
3. **Target Quantization:** The FP16 model is further quantized using `llama-quantize` to the **Q2_K** format. The ultimate goal is to compress the model to fit within a strict **< 500 MB** memory footprint (with a stretch goal of < 250 MB).
4. **Verification:** A final inference pass is run to verify the quantized model retains its tool-calling integrity.

## 🧪 Evaluation (`starter/`)

The `starter/` directory contains everything needed to test and evaluate the model:
- `eval_harness_contract.py`: The official grader interface. Models must implement the `Agent` class and the `predict` method. **Note: No network imports are allowed during evaluation.**
- `teacher_examples.jsonl`: 20 high-quality seed examples of expected inputs and outputs.
- `public_test.jsonl`: A test set to validate model performance locally before submission.
- `tool_schemas.json`: A reference for the expected tool arguments.