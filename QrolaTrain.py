import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
import os

from dotenv import load_dotenv
# 載入環境變數
load_dotenv(override=True)

# 從.env文件中獲取設定
MODEL_FOLDER = os.getenv("MODEL_FOLDER", "models")  # 模型的本地路徑
LLM_NAME = os.getenv("LLM_NAME", "microsoft/DialoGPT-large")  # 模型的本地路徑
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "qrola_output")  # 輸出目錄

RESOURCE_FOLDER=os.getenv("RESOURCE_FOLDER", "Resource")
DATASET_NAME= os.getenv("DATASET_NAME", "dataset.json")

model_path:str = os.path.join(os.path.abspath(MODEL_FOLDER), LLM_NAME)
dataset_path = os.path.join(os.path.abspath(RESOURCE_FOLDER), DATASET_NAME)

# ------------------------------------------
def print_trainable_parameters(model):
    """
    輸出模型的參數
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        # print(f"[輸出模型的參數] {name}: {param.numel()}")
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"[輸出模型的可訓練參數] trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
# ------------------------------------------

# 1. 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 啟用 4-bit 量化
    bnb_4bit_quant_type="nf4",      # 使用 NF4 量化
    bnb_4bit_compute_dtype=torch.float16,  # 計算使用 float16
    bnb_4bit_use_double_quant=False  # 不使用雙重量化
)

# 2. 載入模型和分詞器
model_name = model_path  # 使用 Instruct 版本
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 設置填充標記

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # 自動分配到 GPU
    trust_remote_code=True
)
model.config.use_cache = False  # 訓練時禁用快取

# 3. 準備 LoRA 配置
lora_config = LoraConfig(
    r=16,              # LoRA 秩
    lora_alpha=32,     # 縮放因子
    target_modules=["q_proj", "v_proj"],  # 目標模組（注意力層）
    # target_modules=[
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj",
    #     "gate_proj",
    #     "up_proj",
    #     "down_proj",
    #     "lm_head",
    # ],
    lora_dropout=0.05, # Dropout 比率
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. 啟用量化訓練並應用 LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 5. 載入數據集
print(f"載入數據集: {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")
# 5. 載入並預處理數據集

def formatting_prompts_func(example):
    text = f"[INST] {example['instruction']} [/INST] {example['output']}"
    return text

# 數據預處理：分詞並限制長度
def tokenize_function(example):
    print(f"tokenize_function {str(example)}")
    return tokenizer(
        formatting_prompts_func(example),
        truncation=True,
        max_length=512,  # 在這裡控制最大序列長度
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize_function, batched=False)

# 6. 設置訓練參數
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    warmup_steps=50,
    optim="paged_adamw_8bit"
)

# 7. 初始化 SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    # formatting_func=formatting_prompts_func,  # 已移到 tokenize_function 中處理
)

# 8. 開始訓練
print_trainable_parameters(model)
trainer.train()

# 9. 保存模型
trainer.model.save_pretrained(OUTPUT_DIR + "_save")
tokenizer.save_pretrained(OUTPUT_DIR + "_save")