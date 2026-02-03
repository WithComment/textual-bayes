import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
from multiprocess import Pool

from evaluation.metrics import (accuracy, brier_score, compute_abstention_rate,
                                compute_auc_roc, expected_calibration_error,
                                get_f1_scores)
from utils.checkpointing import save_to_disk
from utils.semantic_uq import SemanticClustering, SemanticUQ

logger = logging.getLogger("Evaluation")


class BaseEvaluator(ABC):
    def __init__(
        self,
        dataloader,
        ensemble,
        transform,
        output_dir,
        answer_regex,
        num_examples=None,
        num_processes=1,
        **model_kwargs,
    ):
        self.dataloader = dataloader
        self.ensemble = ensemble
        self.transform = transform
        self.output_dir = Path(output_dir)
        self.answer_regex = answer_regex
        self.num_examples = num_examples
        self.num_processes = num_processes
        self.model_kwargs = model_kwargs

    @abstractmethod
    def compute_answer(self, example):
        pass

    @abstractmethod
    def compute_metrics(self, results):
        pass

    def evaluate(self):
        # Flatten the effective batch size to be 1 and truncate dataloader
        dataloader = (example for batch in self.dataloader for example in batch)
        dataloader = islice(dataloader, self.num_examples)

        if self.num_processes > 1:
            with Pool(processes=self.num_processes) as pool:
                results = pool.map(self.compute_answer, dataloader)
        else:
            results = [self.compute_answer(example) for example in dataloader]

        save_to_disk(results, self.output_dir / "results.pkl")
        self.compute_metrics(results)


class MultiChoiceEvaluator(BaseEvaluator):
    def __init__(self, *args, choices, **kwargs):
        super().__init__(*args, **kwargs)
        self.choices = choices

    def compute_answer(self, example):
        x, y_true = self.transform(example)
        answers_full = [res["outputs"][-1].value for res in self.ensemble(x, **self.model_kwargs)]

        answers_parsed = []
        for answer in answers_full:
            match = re.search(self.answer_regex, answer)
            answers_parsed.append(match.group(1) if match else None)

        y_probs = np.array(
            [np.mean([answer == letter for answer in answers_parsed]) for letter in self.choices]
        )
        if y_probs.sum() == 0:
            y_probs = np.array([1 / len(self.choices) for _ in self.choices])
        elif y_probs.sum() < 1.0:
            y_probs /= y_probs.sum()

        logger.info(f"Validation Example: {x}\nPrediction: {y_probs}\nAnswer: {y_true}")

        return {
            "y_true": self.choices.index(y_true.value),
            "y_probs": y_probs,
            "answers_full": answers_full,
            "answers_parsed": answers_parsed,
        }

    def compute_metrics(self, results):
        y_true = np.stack([result["y_true"] for result in results])
        y_probs = np.array([result["y_probs"] for result in results])
        y_pred = np.argmax(y_probs, axis=-1)
        y_conf = np.max(y_probs, axis=-1)

        ece = expected_calibration_error(y_true == y_pred, y_conf, num_bins=10)
        acc = accuracy(y_true, y_pred)
        brier = brier_score(y_true, y_probs)
        logger.info(
            f"\nECE: {ece:.2f}\n{'*' * 70}\nACC: {acc:.2f}\n{'*' * 70}\nBRIER: {brier:.2f}\n"
        )


class IntegerMathEvaluator(BaseEvaluator):
    def compute_answer(self, example):
        x, y_true = self.transform(example)
        answers_full = [res["outputs"][-1].value for res in self.ensemble(x, **self.model_kwargs)]

        answers_parsed = []
        for answer in answers_full:
            match = re.search(self.answer_regex, answer)
            answers_parsed.append(int(match.group(1)) if match else None)

        counter = Counter(answers_parsed)
        answers_len = len(answers_parsed)
        ys = {answer: count / answers_len for answer, count in counter.items()}

        logger.info(f"Validation Example: {x}\nPrediction: {ys}\nAnswer: {y_true}")

        return {
            "y_true": int(y_true.value),
            "y_pred": max(ys, key=ys.get),
            "y_conf": max(ys.values()),
            "answers_parsed": answers_parsed,
            "answers_full": answers_full,
        }

    def compute_metrics(self, results):
        y_true = np.stack([result["y_true"] for result in results])
        y_pred = np.array([result["y_pred"] for result in results])
        y_conf = np.array([result["y_conf"] for result in results])

        ece = expected_calibration_error(y_true == y_pred, y_conf, num_bins=10)
        acc = accuracy(y_true, y_pred)
        logger.info(f"\nECE: {ece:.2f}\n{'*' * 70}\nACC: {acc:.2f}\n")


class QasperEvaluator(BaseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sc = SemanticClustering(cuda=True)

    def compute_answer(self, example):
        x, y_true = self.transform(example)
        answers_full = [res["outputs"][-1].value for res in self.ensemble(x, **self.model_kwargs)]

        answers_parsed = []
        for answer in answers_full:
            match = re.search(self.answer_regex, answer)
            answers_parsed.append(match.group(1) if match else "<Not found>")

        semantic_ids = self.sc.get_semantic_ids(
            responses=answers_parsed,
            prompt=example["question"],
            method="llm",
        )
        counter = Counter(semantic_ids)
        answers_len = len(answers_parsed)
        ys = {answer: count / answers_len for answer, count in counter.items()}
        majority_cluster = max(ys, key=ys.get)
        y_pred = answers_parsed[semantic_ids.index(majority_cluster)]
        y_conf = max(ys.values())
        abst_true = int(example["unanswerable"])
        correctness = self.sc.get_correctness_by_llm(y_true, y_pred, example["question"])
        logger.info(f"Validation Example: {x}\nPrediction: {ys}\nAnswer: {y_true}")
        logger.info(f"Answers from model\n" + "\n".join(answers_parsed))
        logger.info(f"Clustering results: {semantic_ids}")
        logger.info(f"Correctness: {correctness}")

        return {
            "y_true": y_true.value,
            "y_pred": y_pred,
            "y_conf": y_conf,
            "answers_parsed": answers_parsed,
            "answers_full": answers_full,
            "abst_pred": -y_conf,
            "abst_true": abst_true,
            "correctness": correctness,
        }

    def compute_metrics(self, results):
        results_answerable = [result for result in results if result["abst_true"] == 0]
        y_true = np.stack([result["y_true"] for result in results_answerable])
        y_pred = np.array([result["y_pred"] for result in results_answerable])
        y_conf = np.array([result["y_conf"] for result in results_answerable])
        correctness = np.array([result["correctness"] for result in results_answerable])
        f1, _ = get_f1_scores(y_true, y_pred)
        acc = np.sum(correctness) / y_true.shape[0]
        ece = expected_calibration_error(correctness, y_conf, num_bins=10)
        logger.info(
            f"\n"
            f"ECE: {ece:.4f}\t{'*' * 70}\n"
            f"Accuracy: {acc:.4f}\t{'*' * 70}\n"
            f"F1: {f1:.4f}\t{'*' * 70}\n"
        )

        # abstention related metrics
        abst_true = np.stack([result["abst_true"] for result in results])
        abst_pred = np.stack([result["abst_pred"] for result in results])
        abst_roc = compute_auc_roc(abst_true, abst_pred)
        logger.info(f"\nAbst Roc: {abst_roc:.4f}\t{'*' * 70}\n")


class SimpleqaEvaluator(BaseEvaluator):
    def __init__(self, *args, **kwargs):
        # TODO: strict matching or LLM eval
        super().__init__(*args, **kwargs)
        self.sc = SemanticClustering(cuda=True)

    def compute_answer(self, example):
        x, y_true = self.transform(example)
        answers_full = [res["outputs"][-1].value for res in self.ensemble(x, **self.model_kwargs)]

        answers_parsed = []
        for answer in answers_full:
            match = re.search(self.answer_regex, answer)
            answers_parsed.append(match.group(1) if match else "<Not found>")

        semantic_ids = self.sc.get_semantic_ids(
            responses=answers_parsed,
            prompt=example["question"],
            method="llm",
        )
        counter = Counter(semantic_ids)
        answers_len = len(answers_parsed)
        ys = {answer: count / answers_len for answer, count in counter.items()}
        majority_cluster = max(ys, key=ys.get)
        y_pred = answers_parsed[semantic_ids.index(majority_cluster)]
        y_conf = max(ys.values())
        equivalence = self.sc.get_correctness_by_llm(y_true, y_pred, example["question"])
        logger.info(f"Validation Example: {x}\nPrediction: {ys}\nAnswer: {y_true}")
        logger.info(f"Answers from model\n" + "\n".join(answers_parsed))
        logger.info(f"Clustering results: {semantic_ids}")

        return {
            "y_true": y_true.value,
            "y_pred": y_pred,
            "y_conf": y_conf,
            "answers_parsed": answers_parsed,
            "answers_full": answers_full,
            "equivalence": equivalence,
        }

    def compute_metrics(self, results):
        y_conf = np.array([result["y_conf"] for result in results])
        equivalence = np.array([result["equivalence"] for result in results])
        acc = np.sum(equivalence) / equivalence.shape[0]
        ece = expected_calibration_error(equivalence, y_conf, num_bins=10)
        logger.info(f"\n" f"ECE: {ece:.4f}\t{'*' * 70}\n" f"Accuracy: {acc:.4f}\t{'*' * 70}\n")
