import evaluate
from transformers import EvalPrediction
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


class ASREvaluator:
    """Evaluator for ASR model using Word Error Rate (WER) metric.
    
    Supports batch_eval_metrics mode for memory-efficient evaluation.
    """

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
        
        # Accumulators for batch_eval_metrics mode
        self._all_predictions = []
        self._all_references = []

    def compute_metrics(self, pred: EvalPrediction, compute_result: bool = True):
        """
        Compute WER metric for model predictions.

        This function supports batch_eval_metrics mode:
        - When compute_result=False: Accumulates batch statistics
        - When compute_result=True: Computes final WER from accumulated data

        Args:
            pred: EvalPrediction object containing predictions and label_ids
            compute_result: If True, compute and return final metrics.
                           If False, accumulate batch data and return empty dict.

        Returns:
            Dictionary containing the WER metric as a percentage (when compute_result=True)
            Empty dictionary (when compute_result=False)
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
            valid_pairs = [(p, l) for p, l in zip(pred_str, label_str) if len(l) > 0]
            if valid_pairs:
                pred_str, label_str = zip(*valid_pairs)
                pred_str, label_str = list(pred_str), list(label_str)
            else:
                pred_str, label_str = [], []

        # Accumulate predictions and references
        self._all_predictions.extend(pred_str)
        self._all_references.extend(label_str)

        if not compute_result:
            # Batch accumulation mode - don't compute final result yet
            return {}

        # Compute final WER from all accumulated data
        if len(self._all_references) == 0:
            wer = 0.0
        else:
            wer = 100 * self.metric.compute(
                predictions=self._all_predictions,
                references=self._all_references
            )

        # Reset accumulators for next evaluation
        self._all_predictions = []
        self._all_references = []

        return {"wer": wer}
