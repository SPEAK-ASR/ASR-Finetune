import evaluate
from transformers import EvalPrediction
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


class ASREvaluator:
    """Evaluator for ASR model using Word Error Rate (WER) metric."""

    def __init__(self, tokenizer, normalize_eval: bool = True):
        """
        Initialize the evaluator.

        Args:
            tokenizer: The tokenizer used for decoding predictions and labels.
            normalize_eval: Whether to normalize text before computing WER (recommended).
        """
        self.tokenizer = tokenizer
        self.metric = evaluate.load("wer")
        self.normalize_eval = normalize_eval
        self.normalizer = BasicTextNormalizer()

    def compute_metrics(self, pred: EvalPrediction):
        """
        Compute WER metric for model predictions.

        This function:
        1. Replaces -100 with pad_token_id in label_ids
        2. Decodes predicted and label ids to strings
        3. Optionally normalizes text for fair comparison
        4. Computes WER between predictions and reference labels

        Args:
            pred: EvalPrediction object containing predictions and label_ids

        Returns:
            Dictionary containing the WER metric as a percentage
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize text for fair WER comparison
        if self.normalize_eval:
            pred_str = [self.normalizer(pred) for pred in pred_str]
            label_str = [self.normalizer(label) for label in label_str]
            # Filter out samples with empty references after normalization
            pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
            label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
