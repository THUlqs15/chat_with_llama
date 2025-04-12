from huggingface_hub import snapshot_download
import os

#yuhuili/EAGLE-mixtral-instruct-8x7B

custom_dir = "/workspace/Butter_QwQ_32B_RPMaster-v0"
os.makedirs(custom_dir, exist_ok=True)


snapshot_download(
    repo_id="trashpanda-org/QwQ-32B-Snowdrop-v0",
    local_dir=custom_dir,
    local_dir_use_symlinks=False,  
    revision="main"  
)

print(f"模型下载完成，存储在 {custom_dir}")
