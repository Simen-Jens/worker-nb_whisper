"""
NbAiLab nb-whisper support. Their repos keep the CTranslate2 weights in ct2/,
while faster-whisper expects tokenizer.json and preprocessor_config.json next
to model.bin (without the latter it assumes 80 mel bins; nb-whisper-large
needs 128), so those are copied in.
"""
import os
import shutil

from huggingface_hub import snapshot_download

MODELS_DIR = "/models"
ROOT_FILES = ("tokenizer.json", "preprocessor_config.json")


def is_nb_whisper(model_name):
    return model_name.startswith("NbAiLab/nb-whisper-")


def nb_whisper_path(repo_id):
    """Download (if needed) and return a directory faster-whisper can load."""
    target = os.path.join(MODELS_DIR, repo_id.split("/", 1)[1])
    ct2_dir = os.path.join(target, "ct2")
    if not all(os.path.exists(os.path.join(ct2_dir, f)) for f in ROOT_FILES):
        snapshot_download(
            repo_id,
            local_dir=target,
            allow_patterns=["ct2/*", *ROOT_FILES],
        )
        for name in ROOT_FILES:
            shutil.copy(os.path.join(target, name), ct2_dir)
    return ct2_dir