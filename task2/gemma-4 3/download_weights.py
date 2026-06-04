import huggingface_hub
huggingface_hub.snapshot_download(repo_id="facebook/nllb-200-distilled-600M", local_dir="weights")
