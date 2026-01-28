"""
Configuration file for UAV Multi-Agent Debate Framework Experiments
"""

# SiliconFlow API Configuration
API_KEY = "sk-pchlhkadtecinrftucdlwcxmahjawylrtdiiwzljdokdevzc"
BASE_URL = "https://api.siliconflow.cn/v1"

# Model Configuration
DEFAULT_MODEL = "Qwen/Qwen3-32B"

# Alternative models available on SiliconFlow:
# - "Qwen/Qwen2.5-7B-Instruct" (faster, lighter)
# - "Qwen/Qwen3-32B" (balanced, recommended)
# - "Pro/Qwen/Qwen2.5-72B-Instruct" (most powerful)
# - "meta-llama/Meta-Llama-3.1-8B-Instruct" (Llama alternative)

# Experiment Settings
MAX_DEBATE_ROUNDS = 3
REQUEST_DELAY = 0.5  # seconds between API calls (rate limiting)

# Dataset Configuration
DATASET_PATH = "../drone_flight_data.json"

# Output Configuration
SAVE_DETAILED_RESULTS = True
OUTPUT_DIR = "."

