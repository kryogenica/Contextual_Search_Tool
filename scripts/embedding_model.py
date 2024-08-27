
import torch
from transformers import DistilBertTokenizer, DistilBertModel, AdamW
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
    
    def get_embeddings(self, text, type):
        if type == 'mean':
            # Function to get BERT embeddings for each text
            # using the mean of each token embedding at berts last layer
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Get the embeddings from the last hidden layer
            last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]

            # Mean pooling: Calculate the mean of all token embeddings (excluding [PAD] tokens)
            attention_mask = inputs['attention_mask']
            masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
            sentence_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        elif type == 'concatenation':
            # Function to get BERT embeddings of a text
            # using the concatenation of the CLS token embeddings for the last four 
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Get the hidden states from all layers
            hidden_states = outputs.hidden_states  # Shape: [layer_count, batch_size, seq_len, hidden_size]

            # Select the last four layers
            last_four_layers = hidden_states[-4:]

            # Concatenate the layers to get a single representation
            # Concatenate the hidden states for each token across the last four layers
            concatenated_hidden_states = torch.cat(last_four_layers, dim=-1)  # Shape: [batch_size, seq_len, hidden_size * 4]

            # Take the embedding for the [CLS] token (assuming it is at index 0)
            sentence_embedding = concatenated_hidden_states[:, 0, :]

        return sentence_embedding.squeeze().numpy()

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
