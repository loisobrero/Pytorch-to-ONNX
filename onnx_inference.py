import torch
import onnxruntime
from transformers import BertTokenizer

def run_onnx_inference(review_text):
    # Load the ONNX model
    onnx_model_path = "movie_review_classifier.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # Print the input names and shapes of the ONNX model
    print("Input names and shapes of the ONNX model:")
    for input_info in ort_session.get_inputs():
        print(f"{input_info.name}: {input_info.shape}")

    # Tokenize the input review text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(review_text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)

    # Run inference with ONNX Runtime
    ort_inputs = {ort_session.get_inputs()[0].name: inputs['input_ids'].numpy()}
    if len(ort_session.get_inputs()) == 2:
        ort_inputs[ort_session.get_inputs()[1].name] = inputs['attention_mask'].numpy()

    ort_outputs = ort_session.run(None, ort_inputs)

    # Get the predicted class
    logits = torch.tensor(ort_outputs[0])
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_class_name = 'positive' if predicted_class == 1 else 'negative'

    print(f"The ONNX model predicted the review as: {predicted_class_name}")
    return predicted_class_name

if __name__ == "__main__":
    review_text = "This is a great movie!"
    run_onnx_inference(review_text)
