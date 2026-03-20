"""Dataset-specific evaluation metrics.

Each function takes (generated, ground_truth_messages, **kwargs) and returns bool.
translation_scores() is separate — returns BLEU/chrF dicts for corpus-level scoring.
"""

import os
import re
import subprocess

import sacrebleu

gsm8k_pattern = re.compile(r"\n#### (.+)$")
spider_pattern = re.compile(r"For db_id:\[(.+)\]")


def gsm8k_eval(generated: str, messages: list[dict], **kwargs) -> bool:
    """GSM8K: regex extract #### answer."""
    gt_match = re.search(gsm8k_pattern, messages[2]["content"])
    gt_answer = None if not gt_match else gt_match.group(1)
    gen_match = re.search(gsm8k_pattern, generated)
    gen_answer = None if not gen_match else gen_match.group(1)
    return gt_answer == gen_answer


def spider_eval(generated: str, messages: list[dict], **kwargs) -> bool:
    """Spider: execute SQL, compare results."""
    spider_path = kwargs.get("spider_path", "")
    db_id_match = re.search(spider_pattern, messages[1]["content"])
    if not db_id_match:
        return False
    db_id = db_id_match.group(1)
    dbfile = os.path.join(spider_path, db_id, db_id + ".sqlite")
    try:
        gen_result = subprocess.run(
            ["sqlite3", dbfile, generated], capture_output=True, text=True,
        ).stdout
        gt_result = subprocess.run(
            ["sqlite3", dbfile, messages[2]["content"]], capture_output=True, text=True,
        ).stdout
    except Exception:
        return False
    return gen_result == gt_result


def nq_open_eval(generated: str, messages: list[dict], **kwargs) -> bool:
    """Natural Questions: substring match from answer list."""
    answer_list = generated.split("; ")
    for answer in answer_list:
        if answer in messages[2]["content"]:
            return True
    return False


def hellaswag_eval(generated: str, messages: list[dict], **kwargs) -> bool:
    """HellaSwag: A/B/C/D comparison."""
    return generated == messages[2]["content"]


def exact_match(generated: str, messages: list[dict], **kwargs) -> bool:
    """Default: exact string match."""
    return generated == messages[2]["content"]


def strip_lang_prefix(text: str) -> str:
    """Remove leading [xx] language tag from translation output."""
    if text.startswith("[") and "]" in text[:6]:
        return text[text.index("]") + 1:].strip()
    return text.strip()


def translation_scores(generated_list: list[str], reference_list: list[str]) -> dict:
    """Compute corpus-level BLEU and chrF for translation pairs.

    Strips [lang] prefixes before scoring. Returns dict with bleu and chrf floats.
    """
    hyps = [strip_lang_prefix(g) for g in generated_list]
    refs = [strip_lang_prefix(r) for r in reference_list]

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])

    return {"bleu": bleu.score, "chrf": chrf.score}


EVAL_REGISTRY = {
    "gsm8k": gsm8k_eval,
    "spider": spider_eval,
    "nq_open": nq_open_eval,
    "hellaswag": hellaswag_eval,
}


def evaluate_sample(generated: str, messages: list[dict], dataset_name: str, **kwargs) -> bool:
    """Dispatch to the right metric based on dataset name prefix."""
    for prefix, fn in EVAL_REGISTRY.items():
        if dataset_name.startswith(prefix):
            return fn(generated, messages, **kwargs)
    return exact_match(generated, messages, **kwargs)
