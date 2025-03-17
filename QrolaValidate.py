from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os, torch
from dotenv import load_dotenv
# 載入環境變數
load_dotenv()

MODEL_FOLDER = os.getenv("MODEL_FOLDER", "models") 
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "qrola_output")  # 輸出目錄，預設為./qrola_output
LLM_NAME = os.getenv("LLM_NAME", "microsoft/DialoGPT-large")  # 預訓練語言模型名稱
model_path: str = os.path.join(os.path.abspath(OUTPUT_DIR), LLM_NAME)
 
 
base_model_path = os.path.join(os.path.abspath(MODEL_FOLDER), LLM_NAME)
lora_weights_path = OUTPUT_DIR + "_save"

# 載入 LoRA 配置
config = PeftConfig.from_pretrained(lora_weights_path)

# 載入原始模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 載入 LoRA 權重
model = PeftModel.from_pretrained(model, lora_weights_path)

# 載入分詞器
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 使用模型
input_text = "<|user|>\n請問台灣有哪些著名的旅遊景點？\n"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs, 
    max_new_tokens=500,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)