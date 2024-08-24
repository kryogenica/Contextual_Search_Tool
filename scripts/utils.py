
import torch

def save_model(model, tokenizer, directory='./fine_tuned_model'):
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)

def load_model(model_class, directory='./fine_tuned_model'):
    model = model_class.from_pretrained(directory)
    tokenizer = model_class.tokenizer_class.from_pretrained(directory)
    return model, tokenizer

def prepare_dataloader(dataset, batch_size=16):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
