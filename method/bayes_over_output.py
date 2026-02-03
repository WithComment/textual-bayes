import logging

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from evaluation.metrics import (accuracy, brier_score,
                                expected_calibration_error)
from utils.logprobs import compute_posterior_probs, compute_prompt_key_logprobs

log = logging.getLogger("Method")


def multi_choice(train_dataloader, method_cfg, num_multi_choices):
    ANSWER_TO_QUESTION_PROMPT_TEMPLATE = method_cfg.prompt_template
    method = method_cfg.logprob_extract_method

    llm = LLM(
        model=method_cfg.llm,
        dtype=method_cfg.dtype,
        max_model_len=method_cfg.max_model_len,
        tensor_parallel_size=method_cfg.tensor_parallel_size,
        gpu_memory_utilization=method_cfg.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(temperature=method_cfg.temperature, seed=method_cfg.seed)

    posterior_prob_list, labels = [], []
    for batch in tqdm(train_dataloader):
        likelihood_logprobs = []
        for i in range(num_multi_choices):
            prompt_dicts = []
            for prompt in batch:
                prompt_dicts.append(
                    {"question": prompt["question"], "answer": prompt["answers"][i]}
                )
            likelihood_logprobs.append(
                compute_prompt_key_logprobs(
                    ANSWER_TO_QUESTION_PROMPT_TEMPLATE,
                    prompt_dicts,
                    "question",
                    llm,
                    sampling_params,
                    method=method,
                )
            )

        # Group by question
        likelihood_logprobs = np.asarray(list(zip(*likelihood_logprobs)))

        # Uniform prior
        prior_logprobs = np.ones_like(likelihood_logprobs) * np.log(1 / num_multi_choices)

        # posterior probs from logprobs
        posterior_probs = compute_posterior_probs(likelihood_logprobs, prior_logprobs)
        posterior_prob_list.append(posterior_probs)

        # Ground truth
        labels.append(np.asarray([prompt["label"] for prompt in batch]))

    # Aggregate over the whole dataset
    posterior_probs = np.concatenate(posterior_prob_list)
    y_true = np.concatenate(labels)
    y_pred = np.argmax(posterior_probs, axis=-1)  # Model's top answer
    y_conf = np.max(posterior_probs, axis=-1)  # Model confidence in the top answer

    # Compute metrics
    ece = expected_calibration_error(y_true, y_pred, y_conf, num_bins=10)
    acc = accuracy(y_true, y_pred)
    brier = brier_score(y_true, posterior_probs)
    result = {"ECE": ece, "ACC": acc, "BRIER": brier}

    log.info(f"\nECE: {ece:.2f}\n{'*' * 70}\nACC: {acc:.2f}\n{'*' * 70}\nBRIER: {brier:.2f}\n")

    return result
