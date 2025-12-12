import json
import os

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "text_model": "qwen-plus",
    "vl_model": "qwen-vl-plus",
    "enable_tools": True
}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            return {**DEFAULT_CONFIG, **config}
    except:
        return DEFAULT_CONFIG

def save_config(config_dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        return True
    except Exception as e:
        print(f"Save config failed: {e}")
        return False