import torch
from transformers import BertTokenizer, BertForSequenceClassification

class MovieReviewClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    def classify_review(self, review_text):
        inputs = self.tokenizer(review_text, return_tensors='pt')

        # Convert the input tensor to LongTensor explicitly
        inputs = {key: val.to(torch.long) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_class_name = 'positive' if predicted_class == 1 else 'negative'

        print(f"The model predicted the review as: {predicted_class_name}")
        return predicted_class_name