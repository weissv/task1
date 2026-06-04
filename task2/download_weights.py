"""Download NLLB-200 base weights for fine-tuning.
After fine-tuning in the notebook, the weights/ folder will contain
the fine-tuned model instead of the base model.
"""
import huggingface_hub
huggingface_hub.snapshot_download(
    repo_id="facebook/nllb-200-distilled-600M",
    local_dir="weights"
)
print("✅ Base NLLB-200-distilled-600M weights downloaded to ./weights")
print("Now run the training notebook to fine-tune on Abkhazian data.")
