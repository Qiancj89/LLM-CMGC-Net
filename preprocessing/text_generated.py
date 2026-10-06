import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
from tqdm import tqdm
import gc
import os
import traceback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()

MODEL_REGISTRY = {
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen-14B": "Qwen/Qwen2.5-14B-Instruct",
    "GLM-9B": "THUDM/glm-4-9b-chat",
    "Huatuo-7B": "FreedomIntelligence/HuatuoGPT2-7B"
}

CURRENT_MODEL_KEY = "Huatuo-7B"

INPUT_FILE = r"TestData.xlsx"

OUTPUT_CSV = f"clinical_summary_{CURRENT_MODEL_KEY}_testdata.csv"
MODEL_PATH = MODEL_REGISTRY[CURRENT_MODEL_KEY]

print(f"正在以 4-bit 量化模式加载 {CURRENT_MODEL_KEY} 到 RTX 4090，请稍候...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True
)
model.eval()


def safe_str(val):
    if pd.isna(val):
        return "未查"
    val_str = str(val).strip()
    if val_str == "" or val_str.upper() in ["-", "NULL", "N/A", "NA", "无", "未测"]:
        return "未查"
    if isinstance(val, float):
        return f"{val:g}"
    return val_str


def build_prompt(row):
    age = safe_str(row.get('年龄'))
    meno_raw = str(row.get('绝经状态')).strip()
    menopausal = "已绝经" if meno_raw in ['1', '1.0', '是'] else "未绝经" if meno_raw in ['0', '0.0', '否'] else "未知"
    location = safe_str(row.get('部位'))

    he4 = safe_str(row.get('HE4'))
    afp = safe_str(row.get('AFP'))
    cea = safe_str(row.get('CEA'))
    ca125 = safe_str(row.get('CA125'))
    ca153 = safe_str(row.get('CA153'))
    ca199 = safe_str(row.get('CA199'))
    ca724 = safe_str(row.get('CA724'))
    nse = safe_str(row.get('NSE'))
    cy211 = safe_str(row.get('CY211'))

    prompt = f"""你是一位严谨的妇科肿瘤临床医生。请根据以下患者的基础信息和血液肿瘤标志物检验结果，生成一段客观的【临床与检验摘要】（约100字）。

【硬性规则（防幻觉警告）】
1. 必须严格参照以下正常值范围判断指标是否升高：
   - CA125 正常 < 35 U/mL
   - HE4 正常 (绝经前 < 70 pmol/L, 绝经后 < 140 pmol/L)
   - CEA 正常 < 5 ng/mL
   - AFP 正常 < 20 ng/mL
   - CA153 正常 < 31.3 U/mL
   - CA199 正常 < 37 U/mL
   - CA724 正常 < 6.9 U/mL
2. 对于异常升高的指标，必须给出具体数值并提示其临床意义。
3. 对于在正常范围内的指标，不要罗列具体数值，只需统一概括为“处于正常范围”。
4. 对于没有任何数值的指标，统一描述为“未查”。
5. 绝对不要描述超声影像特征（如大小、血流）。
6. 保持客观，使用“提示”、“可能”，绝不要下最终确诊结论。

【患者输入数据】
- 基础信息：该患者，年龄 {age} 岁，绝经状态：{menopausal}。肿瘤位于：{location}。
- 检验数据：
  CA125={ca125}, HE4={he4}, CEA={cea}, AFP={afp}, CA153={ca153}, CA199={ca199}, CA724={ca724}, NSE={nse}, CY211={cy211}。
"""
    return prompt


def generate_medical_narrative_local(prompt):
    messages = [
        {"role": "system", "content": "你是一个严谨的医学大模型，擅长撰写妇科肿瘤的病史与检验报告摘要。"},
        {"role": "user", "content": prompt}
    ]

    with torch.no_grad():

        if "huatuo" in CURRENT_MODEL_KEY.lower():
            system_role = messages[0]["content"]
            user_content = messages[1]["content"]

            huatuo_prompt = f"<问>：{system_role} 请务必极其简短地总结，不要长篇大论。\n\n{user_content}\n<答>："

            model_inputs = tokenizer(huatuo_prompt, return_tensors="pt").to(model.device)

            eos_id = tokenizer.eos_token_id
            if isinstance(eos_id, list):
                eos_id = eos_id[0]

            generated_ids = model.generate(
                input_ids=model_inputs.input_ids,
                max_new_tokens=512,
                temperature=0.1,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=eos_id,
                eos_token_id=eos_id,
                use_cache=False
            )

            input_length = model_inputs.input_ids.shape[1]
            new_tokens = generated_ids[0][input_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()

        elif "glm" in CURRENT_MODEL_KEY.lower():

            model_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False
            )

            input_length = model_inputs["input_ids"].shape[1]
            new_tokens = generated_ids[0][input_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()

        else:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            input_length = model_inputs.input_ids.shape[1]
            new_tokens = generated_ids[0][input_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()


def process_dataset():
    print(f"读取原始数据集: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE, sheet_name="Sheet1")
    df.columns = [str(col).strip() for col in df.columns]

    if 'Clinical_Summary' not in df.columns:
        df['Clinical_Summary'] = None

    pending_indices = df[df['Clinical_Summary'].isnull()].index
    print(f"共需处理 {len(pending_indices)} 条数据，当前使用模型: {CURRENT_MODEL_KEY}")

    for idx in tqdm(pending_indices, desc="生成文本特征中"):
        row = df.loc[idx]
        prompt = build_prompt(row)

        try:
            narrative = generate_medical_narrative_local(prompt)
            df.at[idx, 'Clinical_Summary'] = narrative
        except Exception as e:
            print(f"\n================ 致命错误发生在第 {idx} 行 ================")
            traceback.print_exc()
            print("=====================================================")
            break

        if idx > 0 and idx % 5 == 0:
            df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            torch.cuda.empty_cache()
            gc.collect()

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n全部临床摘要特征提取完成！结果已保存在: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_dataset()
