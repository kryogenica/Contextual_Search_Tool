
import torch
from transformers import DistilBertTokenizer, DistilBertModel, AdamW
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
    
    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.detach().numpy()

    def fine_tune_model(self, train_dataset, epochs=3, learning_rate=5e-5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.model.train()
        
        for epoch in range(epochs):
            for batch in train_dataset:
                inputs = self.tokenizer(batch['text'], return_tensors='pt', truncation=True, padding=True)
                labels = batch['labels']
                
                outputs = self.model(**inputs)
                loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self.model

    def save_model(self, save_directory='./fine_tuned_distilbert'):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def load_fine_tuned_model(self, load_directory='./fine_tuned_distilbert'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(load_directory)
        self.model = DistilBertModel.from_pretrained(load_directory)
