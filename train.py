"""Unified training entry point.

Usage:
    python train.py --config configs/experiments/replicate_stp_synth.toml
    python train.py --config configs/strategies/stp.toml --set data.train_file=datasets/synth_train.jsonl
    python train.py --config configs/experiments/replicate_stp_synth.toml --set strategy.name=regular
"""

import math
import os
import shutil
import sys
import time

import torch
from transformers import DataCollatorForLanguageModeling, TrainingArguments

from llm_jepa.callbacks import CSVLossCallback, ProfilerFLOPCallback, TranslationEvalCallback
from llm_jepa.config import load_settings
from llm_jepa.data import load_and_prepare_dataset
from llm_jepa.models import setup_model_and_tokenizer
from llm_jepa.strategies import get_trainer_class
from llm_jepa.utils import is_rank_zero, setup_logging


def main():
    settings = load_settings()
    setup_logging(settings.get("debug.level", "INFO"))

    strategy_name = settings.strategy.name
    if is_rank_zero():
        print(f"=== LLM-JEPA Training: strategy={strategy_name} ===")
        print(f"Model: {settings.model.name}")

    # --- Distributed setup ---
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        if is_rank_zero():
            print(f"Distributed: world_size={world_size}, local_rank={local_rank}")

    # --- Model & tokenizer ---
    # Always add mask token to match original stp.py behavior (affects embedding
    # layer size and therefore random init, even when mask token isn't used).
    tokenizer_name = settings.get("model.tokenizer_name", "") or None
    model, tokenizer = setup_model_and_tokenizer(
        settings.model.name,
        use_lora=settings.model.use_lora,
        lora_rank=settings.model.lora_rank,
        pretrain=settings.model.pretrain,
        seed=settings.training.seed,
        linear_predictor=settings.strategy.linear_predictor,
        load_lp=settings.model.load_lp,
        add_mask_token=True,
        tokenizer_name=tokenizer_name,
    )

    # --- Dataset ---
    data_strategy = strategy_name
    if strategy_name == "regular":
        data_strategy = "regular"

    if settings.data.train_file:
        train_dataset = load_and_prepare_dataset(
            settings.data.train_file, tokenizer, settings.model.name,
            max_length=settings.data.max_length,
            strategy=data_strategy,
            predictors=settings.strategy.predictors,
            train_all=settings.data.train_all,
            plain=settings.data.plain,
            front_pred=settings.get("strategy.jepa.front_pred", False),
            reverse_pred=settings.get("strategy.jepa.reverse_pred", False),
            plain_jepa=settings.get("strategy.jepa.plain_jepa", False),
            same_predictor=settings.get("strategy.jepa.same_predictor", False),
        )
        eval_dataset = None
        if settings.data.eval_file:
            eval_dataset = load_and_prepare_dataset(
                settings.data.eval_file, tokenizer, settings.model.name,
                max_length=settings.data.max_length,
                strategy=data_strategy,
                train_all=settings.data.train_all,
                plain=settings.data.plain,
                front_pred=settings.get("strategy.jepa.front_pred", False),
                reverse_pred=settings.get("strategy.jepa.reverse_pred", False),
                plain_jepa=settings.get("strategy.jepa.plain_jepa", False),
                same_predictor=settings.get("strategy.jepa.same_predictor", False),
            )
    elif settings.data.data_file:
        full_dataset = load_and_prepare_dataset(
            settings.data.data_file, tokenizer, settings.model.name,
            max_length=settings.data.max_length,
            strategy=data_strategy,
            predictors=settings.strategy.predictors,
            train_all=settings.data.train_all,
            plain=settings.data.plain,
            front_pred=settings.get("strategy.jepa.front_pred", False),
            reverse_pred=settings.get("strategy.jepa.reverse_pred", False),
            plain_jepa=settings.get("strategy.jepa.plain_jepa", False),
            same_predictor=settings.get("strategy.jepa.same_predictor", False),
        )
        if settings.data.eval_split > 0:
            split = full_dataset.train_test_split(
                test_size=settings.data.eval_split,
                seed=settings.data.split_seed,
                shuffle=True,
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = full_dataset
            eval_dataset = None
    else:
        raise ValueError("Must provide data.train_file or data.data_file")

    max_train_samples = settings.get("data.max_train_samples", -1)
    if max_train_samples > 0 and len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))

    max_eval_samples = settings.get("data.max_eval_samples", -1)
    if max_eval_samples > 0 and eval_dataset and len(eval_dataset) > max_eval_samples:
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    if is_rank_zero():
        print(f"Train samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Eval samples: {len(eval_dataset)}")

    # --- Dry run ---
    if settings.get("dry_run", False):
        print("Dry run complete. First batch:")
        for key in train_dataset.column_names:
            print(f"  {key}: {type(train_dataset[0][key])}")
        return

    # --- Training args ---
    output_dir = os.path.abspath(settings.get("training.output_dir", "./output"))
    eval_steps = settings.training.eval_steps
    save_steps = len(train_dataset) // (world_size * settings.training.batch_size * settings.training.grad_accum)
    save_steps = max(save_steps, 1)
    num_epochs = settings.training.num_epochs

    if settings.training.same_flop:
        ratio = settings.get("strategy.jepa.ratio", -1.0)
        if ratio > 0.0:
            save_steps = int(save_steps / (1 + ratio))
            num_epochs = int(math.ceil(num_epochs / (1 + ratio)))
        elif settings.get("strategy.jepa.additive_mask", False):
            save_steps = save_steps // 2
            num_epochs = int(math.ceil(num_epochs / 2))
        elif strategy_name != "regular":
            save_steps = save_steps // 3
            num_epochs = int(math.ceil(num_epochs / 3))

    if is_rank_zero():
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

        # Tee stdout/stderr to a log file
        log_file = open(os.path.join(output_dir, "train.log"), "w")

        class _Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()
            def flush(self):
                for s in self.streams:
                    s.flush()

        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)

    enable_save = settings.get("training.enable_save", True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=settings.training.batch_size,
        per_device_eval_batch_size=settings.training.batch_size,
        gradient_accumulation_steps=settings.training.grad_accum,
        lr_scheduler_type=settings.get("training.lr_scheduler_type", "linear") if not settings.training.constant_lr else "constant",
        warmup_steps=settings.get("training.warmup_steps", 0),
        weight_decay=settings.get("training.weight_decay", 0.0),
        learning_rate=settings.training.learning_rate,
        num_train_epochs=num_epochs,
        eval_strategy="no",
        save_strategy="steps" if enable_save else "no",
        save_steps=save_steps if enable_save else None,
        save_total_limit=num_epochs * 4 if enable_save else None,
        logging_dir=f"{output_dir}/logs",
        logging_steps=eval_steps,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl" if world_size > 1 else None,
        fsdp="",
        fsdp_config={},
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
        tf32=False,
        seed=settings.training.seed,
        data_seed=settings.training.seed,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None,
    )

    # --- Callbacks ---
    callbacks = []
    if settings.training.track_flop:
        callbacks.append(ProfilerFLOPCallback())
    if is_rank_zero():
        callbacks.append(CSVLossCallback(output_dir))
        if settings.data.eval_file:
            callbacks.append(TranslationEvalCallback(
                eval_file=settings.data.eval_file,
                tokenizer=tokenizer,
                model_name=settings.model.name,
                max_samples=settings.get("training.eval_gen_samples", 50),
                eval_every=settings.get("training.eval_gen_every", 5),
                max_length=settings.data.max_length,
                max_new_tokens=settings.get("evaluation.max_new_tokens", 128),
                plain=settings.data.plain,
                output_dir=output_dir,
            ))

    # --- Trainer ---
    trainer_cls = get_trainer_class(strategy_name)

    if strategy_name == "regular":
        trainer = trainer_cls(
            model=model, args=training_args,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            processing_class=tokenizer, data_collator=data_collator,
            callbacks=callbacks,
        )
    elif strategy_name == "jepa":
        trainer = trainer_cls(
            model=model, args=training_args,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            processing_class=tokenizer, data_collator=data_collator,
            callbacks=callbacks,
            lbd=settings.strategy.lambda_,
            gamma=settings.strategy.gamma,
            last_token=settings.strategy.last_token,
            loss_type=settings.strategy.loss_type,
            lbd_warmup=settings.strategy.lbd_warmup,
            min_lbd=settings.strategy.min_lbd,
            linear_predictor=settings.strategy.linear_predictor,
            additive_mask=settings.strategy.jepa.additive_mask,
            jepa_ratio=settings.strategy.jepa.ratio,
            avg_encoding=settings.strategy.jepa.avg_encoding,
        )
    elif strategy_name == "stp":
        stp = settings.strategy.stp
        trainer = trainer_cls(
            model=model, args=training_args,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            processing_class=tokenizer, data_collator=data_collator,
            callbacks=callbacks,
            lbd=settings.strategy.lambda_,
            gamma=settings.strategy.gamma,
            last_token=settings.strategy.last_token,
            loss_type=settings.strategy.loss_type,
            lbd_warmup=settings.strategy.lbd_warmup,
            min_lbd=settings.strategy.min_lbd,
            linear_predictor=settings.strategy.linear_predictor,
            linear_mode=stp.linear_mode or None,
            span_max_length=stp.span_max_length,
            span_times=stp.span_times,
            span_layer=stp.span_layer,
            span_zero=stp.span_zero,
            span_e2e=stp.span_e2e,
            span_all=stp.span_all,
            span_draw_both=stp.span_draw_both,
            span_uniform=stp.span_uniform,
            length_adjustment=stp.length_adjustment or None,
            curvature_sign=stp.curvature_sign,
            random_span_mask=stp.random_span_mask,
            random_span_mask_recover=stp.random_span_mask_recover,
            additive_mask=settings.get("strategy.jepa.additive_mask", False),
            jepa_ratio=settings.get("strategy.jepa.ratio", -1.0),
            avg_encoding=settings.get("strategy.jepa.avg_encoding", False),
        )

    # --- Train ---
    if is_rank_zero():
        print("Starting training...")
    trainer.train()

    # --- Save ---
    if is_rank_zero():
        retry = 3
        while retry > 0:
            try:
                if settings.model.use_lora:
                    merged = model.merge_and_unload()
                    merged.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                else:
                    trainer.save_model()
                    trainer.save_state()
                    tokenizer.save_pretrained(output_dir)
                break
            except Exception as e:
                print(f"Save failed: {e}")
                retry -= 1
                if retry <= 0:
                    raise
                time.sleep(10)

        print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
