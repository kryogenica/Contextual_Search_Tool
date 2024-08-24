
import numpy as np

class SearchTool:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.scene_embeddings = {}
        self.dialogue_embeddings = {}

    def embed_scenes_and_dialogues(self, scenes):
        for i, scene in enumerate(scenes):
            descriptions, dialogues = self.embedding_model.separate_scene(scene)
            scene_embedding = self.embedding_model.get_embeddings(descriptions)
            self.scene_embeddings[i] = scene_embedding
            self.dialogue_embeddings[i] = self.embedding_model.get_embeddings(dialogues)

    def search_dialogue(self, query_text, character_name=None):
        query_embedding = self.embedding_model.get_embeddings(query_text)
        similarities = {}

        for i, dialogue_embedding in self.dialogue_embeddings.items():
            similarity = self.cosine_similarity(query_embedding, dialogue_embedding)
            similarities[i] = similarity

        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        return sorted_similarities

    def cosine_similarity(self, emb1, emb2):
        emb1 = emb1 / np.linalg.norm(emb1, axis=-1, keepdims=True)
        emb2 = emb2 / np.linalg.norm(emb2, axis=-1, keepdims=True)
        return np.dot(emb1, emb2.T).squeeze()

    def compare_embeddings(self, query_embedding, target_embedding):
        return self.cosine_similarity(query_embedding, target_embedding)
