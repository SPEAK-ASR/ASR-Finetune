"""Evaluator for the joint pipeline. Computes WER on the post-processor's final
clean-text output, using the ByT5 (or whichever) post-processor tokenizer for
decoding.

Follows the batch_eval_metrics pattern from src/components/evaluator.py.
"""

import evaluate
from transformers import EvalPrediction
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


class JointEvaluator:
    """Computes WER between the post-processor's predictions and the clean labels."""

    def __init__(self, postproc_tokenizer, normalize_eval: bool = True):
        self.tokenizer = postproc_tokenizer
        self.metric = evaluate.load("wer")
        self.normalize_eval = normalize_eval
        self.normalizer = BasicTextNormalizer()

        self._all_predictions: list[str] = []
        self._all_references: list[str] = []

    def compute_metrics(self, pred: EvalPrediction, compute_result: bool = True):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # ``label_ids`` here is our ``clean_labels`` (-100 masked).
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if self.normalize_eval:
            pred_str = [self.normalizer(p) for p in pred_str]
            label_str = [self.normalizer(l) for l in label_str]
            valid_pairs = [(p, l) for p, l in zip(pred_str, label_str) if len(l) > 0]
            if valid_pairs:
                pred_str, label_str = zip(*valid_pairs)
                pred_str, label_str = list(pred_str), list(label_str)
            else:
                pred_str, label_str = [], []

        self._all_predictions.extend(pred_str)
        self._all_references.extend(label_str)

        if not compute_result:
            return {}

        if len(self._all_references) == 0:
            wer = 0.0
        else:
            wer = 100 * self.metric.compute(
                predictions=self._all_predictions,
                references=self._all_references,
            )

        self._all_predictions = []
        self._all_references = []

        return {"wer": wer}
