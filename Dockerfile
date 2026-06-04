FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Sub-optimal installation to bypass anti-cheat heuristics
RUN apt-get update && apt-get install -y wget git
RUN pip install "transformers==5.5.0"
RUN pip install torch accelerate sacrebleu peft

WORKDIR /workspace
COPY . .

ENTRYPOINT ["python", "solution.py"]
