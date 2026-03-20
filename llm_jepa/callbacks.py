"""Training callbacks."""

import csv
import json
import logging
import os

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import TrainerCallback

logger = logging.getLogger("llm_jepa")


class ProfilerFLOPCallback(TrainerCallback):
    """Profile FLOPs for the first N training steps."""

    def __init__(self, profile_steps: int = 10):
        self.profile_steps = profile_steps
        self.total_flops = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step < self.profile_steps:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
            )
            self.profiler.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step < self.profile_steps:
            self.profiler.__exit__(None, None, None)
            events = self.profiler.key_averages()
            step_flops = sum(event.flops for event in events if event.flops > 0)
            self.total_flops += step_flops
            if torch.cuda.current_device() == 0:
                print(f"Step {state.global_step}: FLOPs: {step_flops:,.0f}")


class CSVLossCallback(TrainerCallback):
    """Write training metrics to a CSV file at each logging step."""

    def __init__(self, output_dir: str):
        self.csv_path = os.path.join(output_dir, "training_log.csv")
        self._header_written = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        row = {
            "step": state.global_step,
            "epoch": round(logs.get("epoch", 0), 4),
            "loss": round(logs["loss"], 6),
            "learning_rate": logs.get("learning_rate", 0),
        }
        # Include aux_loss / lm_loss if the strategy logs them
        for key in ("aux_loss", "lm_loss"):
            if key in logs:
                row[key] = round(logs[key], 6)

        write_header = not self._header_written
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


class TranslationEvalCallback(TrainerCallback):
    """Periodically generate translations on eval subset and compute BLEU/chrF.

    Runs every `eval_every` logging steps (not training steps — this fires
    inside on_log, so it aligns with the same cadence as loss reporting).
    """

    def __init__(
        self,
        eval_file: str,
        tokenizer,
        model_name: str,
        max_samples: int = 50,
        eval_every: int = 5,
        max_length: int = 512,
        max_new_tokens: int = 128,
        plain: bool = False,
        output_dir: str = ".",
    ):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.plain = plain
        self.eval_every = eval_every
        self.csv_path = os.path.join(output_dir, "eval_scores.csv")
        self._log_count = 0
        self._header_written = False

        # Load raw eval examples once
        with open(eval_file) as f:
            all_examples = [json.loads(line) for line in f]
        self.examples = all_examples[:max_samples]
        logger.info(f"TranslationEvalCallback: loaded {len(self.examples)} eval examples")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        self._log_count += 1
        if self._log_count % self.eval_every != 0:
            return

        model = kwargs.get("model")
        if model is None:
            return

        self._run_eval(model, state.global_step, logs.get("epoch", 0))

    def _run_eval(self, model, step, epoch):
        from llm_jepa.evaluation.evaluate import format_conversation, generate_response
        from llm_jepa.evaluation.metrics import translation_scores
        from llm_jepa.models import get_adapter

        adapter = get_adapter(self.model_name)
        model.eval()

        generated_list = []
        reference_list = []

        for ex in self.examples:
            messages = ex["messages"]
            full_messages = adapter.get_messages(messages)
            prompt = format_conversation(
                full_messages, self.tokenizer, plain=self.plain,
            )
            with torch.no_grad():
                generated = generate_response(
                    model, self.tokenizer, prompt,
                    self.max_length, self.max_new_tokens,
                )
            generated_list.append(generated)
            reference_list.append(messages[2]["content"])

        scores = translation_scores(generated_list, reference_list)
        print(f"[Step {step}] Eval BLEU: {scores['bleu']:.2f}, chrF: {scores['chrf']:.2f}")

        row = {
            "step": step,
            "epoch": round(epoch, 4),
            "bleu": round(scores["bleu"], 2),
            "chrf": round(scores["chrf"], 2),
        }
        write_header = not self._header_written
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

        model.train()
