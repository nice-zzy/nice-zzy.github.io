from minimind.model import LMModel
from minimind.utils import load_dataset, evaluate_model
from peft import get_peft_model, LoraConfig
from trl import DPOTrainer

# 加载Dense模型
dense_model = LMModel.from_pretrained("minimind-v1")

# 加载MoE模型
moe_model = LMModel.from_pretrained("minimind-v1-moe")

# 应用LoRA
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
lora_model = get_peft_model(dense_model, lora_config)

# 应用DPO
dpo_trainer = DPOTrainer(model=dense_model)
dpo_trainer.train()

# 评估模型
datasets = {
    "C-Eval": load_dataset("ceval"),
    "CMMLU": load_dataset("cmmlu"),
    "MMLU": load_dataset("mmlu")
}

for dataset_name, dataset in datasets.items():
    print(f"Evaluating on {dataset_name}:")
    print("Dense model:", evaluate_model(dense_model, dataset))
    print("MoE model:", evaluate_model(moe_model, dataset))
    print("LoRA model:", evaluate_model(lora_model, dataset))
    print("DPO model:", evaluate_model(dpo_trainer.model, dataset))
