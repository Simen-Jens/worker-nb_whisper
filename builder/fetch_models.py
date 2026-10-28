"""Bake the default model (MODEL_ID) into the image so cold starts skip the download."""
import os

from faster_whisper.utils import download_model
from nb_models import is_nb_whisper, nb_whisper_path

model_id = os.environ.get("MODEL_ID", "base")
print(f"Downloading {model_id}...")
if is_nb_whisper(model_id):
    nb_whisper_path(model_id)
else:
    download_model(model_id)
print(f"Finished downloading {model_id}.")