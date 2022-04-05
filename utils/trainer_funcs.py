from transformers import AutoModelForSequenceClassification

def model_init(checkpoint):
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, return_dict=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)