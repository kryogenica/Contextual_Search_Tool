

import numpy as np
import pandas as pd
import pickle
import time
import re
import os

class SearchTool:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.action_embeddings = {}
        self.dialogue_embeddings = {}

    def clean_text(self, text):
        # Remove non-printable characters (including \xad and \n)
        clean_text = ''.join(c if c.isprintable() else ' ' for c in text)
        # Remove any addition white spaces found between strings.
        clean_text = re.sub(' +', ' ', clean_text)
        # Remove any leading or trailing white spaces.
        clean_text = clean_text.strip()
        return clean_text

    def embed_actions_and_dialogues(self, scene_system):
        self.script_location = scene_system.script_location
        start_time = time.time()

        self.action_embeddings = {}
        self.dialogue_embeddings = {}

        for scene_number in range(len(scene_system.scenes)):
            actions, dialogues = scene_system.scene_distiller(scene_number)
            self.action_embeddings[scene_number] = [self.embedding_model.get_embeddings( self.clean_text(action.lower()), 'mean' ) for action in actions]
            if dialogues:
                self.dialogue_embeddings[scene_number] = [self.embedding_model.get_embeddings( self.clean_text(chat.get(next(iter(chat))).lower()), 'mean' ) for chat in dialogues]
            else:
                self.dialogue_embeddings[scene_number] = []

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Time taken to embed actions and dialogoue of each scene across all the movie script: {elapsed_time:.2f} seconds")

    def search_dialogue(self, query_text, embedding_type='mean', character_name=None):
        query_embedding = self.embedding_model.get_embeddings(query_text, embedding_type)
        similarities = {}

        for i, dialogue_embedding in self.dialogue_embeddings.items():
            similarity = self.cosine_similarity(query_embedding, dialogue_embedding)
            similarities[i] = similarity

        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        return sorted_similarities
    

    def search_dialogue(self, query_text, scene_system, embedding_type = 'mean', character_name=None):
        query_embedding =  self.embedding_model.get_embeddings( self.clean_text(query_text.lower()), embedding_type )
        most_similar_dialogues = []

        for scene_number, dialogue_embeddings in self.dialogue_embeddings.items():
            _, dialogues = scene_system.scene_distiller(scene_number)
            for i, dialogue_embedding in enumerate(dialogue_embeddings):
                similarity = self.cosine_similarity([query_embedding], [dialogue_embedding])[0][0]
                
                if character_name:
                    # Check if the character name matches
                    for character, dialogue in dialogues[i].items():
                        if character_name.lower() in character.lower():
                            most_similar_dialogues.append((similarity, scene_number, i, character, dialogue))

                else:
                    # No character name filter
                    for character, dialogue in dialogues[i].items():
                        most_similar_dialogues.append((similarity, scene_number, i, character, dialogue))

        # Sort dialogues by similarity in descending order
        most_similar_dialogues.sort(reverse=True)

        # Take the top 5 most similar dialogues
        most_similar_dialogues = most_similar_dialogues[:5]

        # Convert to a pandas DataFrame for nice grid output
        results_df = pd.DataFrame(most_similar_dialogues, columns=["Similarity", "Scene", "Dialogue Index", "Character", "Dialogue"])

        # Clean diaologue text
        results_df['Dialogue'] = results_df['Dialogue'].apply(self.clean_text)

        # Print the DataFrame in a nice grid format
        return results_df
    
    def search_actions(self, query_text, scene_system, embedding_type = 'mean', top_n=5):
        """
        Finds the most similar actions to a given query text.

        Args:
            query_text: The text to find similar scenes for.
            scene_system: A dictionary mapping scene numbers to lists of action embeddings.
            embedding_type: type of embedding being used.
            top_n: The number of most similar scenes to return.

        Returns:
            A list of tuples (scene_index, desc_index, similarity_score, action_text) representing the top_n most similar scenes.
        """
        query_embedding =  self.embedding_model.get_embeddings( self.clean_text(query_text.lower()), embedding_type )

        results = []
        for scene_index, act_embeddings in self.action_embeddings.items():
            for act_index, act_embedding in enumerate(act_embeddings):
                similarity = self.cosine_similarity([query_embedding], [act_embedding])[0][0]
                results.append((similarity, scene_index, act_index))

        # Sort results by similarity score in descending order
        results.sort(key=lambda x: x[0], reverse=True)
        # Return the top_n results
        results = results[:top_n]

        # Appending the action text to each respective tuple
        for i, _ in enumerate(results):
            actions, _ = scene_system.scene_distiller(results[i][1])
            results[i] = results[i] + (self.clean_text(actions[results[i][2]]),)

        # Convert to a pandas DataFrame for nice grid output
        results_df = pd.DataFrame(results, columns=["Similarity", "Scene", "Action Index", "Text"])
        
        return results_df

    def cosine_similarity(self, X, Y=None):
        """Compute cosine similarity between samples in X and Y using only NumPy."""
        
        # Normalize the input matrices
        def normalize(X):
            if not isinstance(X, np.ndarray):
                X = np.array(X)  # Ensure X is a NumPy array
                
            if X.ndim != 2:
                print(X)
                print(X.ndim)
                raise ValueError("Input must be a 2D array or matrix.")
                
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            # Handle the case where the norm is zero to avoid division by zero
            norms[norms == 0] = 1
            return X / norms
        
        # Convert X and Y to NumPy arrays if necessary
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if Y is not None and not isinstance(Y, np.ndarray):
            Y = np.array(Y)

        # If Y is not provided, set it to be X (pairwise similarities within X)
        if Y is None:
            Y = X

        # Normalize the input matrices X and Y
        X_normalized = normalize(X)
        Y_normalized = normalize(Y)
        
        # Compute the cosine similarity as the dot product between X and Y
        similarity_matrix = np.dot(X_normalized, Y_normalized.T)
        
        return similarity_matrix

    def compare_embeddings(self, query_embedding, target_embedding):
        return self.cosine_similarity(query_embedding, target_embedding)
    
    def save_embeddings(self, file_path):
            # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        # Save the embeddings, model information, and script location to a file
        data = {
            'action_embeddings': self.action_embeddings,
            'dialogue_embeddings': self.dialogue_embeddings,
            'embedding_model_name': self.embedding_model.model.name_or_path,  # model name to be used to load later
            'script_location': self.script_location
        }
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data saved to {file_path}")

    def load_embeddings(self, file_path):
        from scripts.embedding_model import EmbeddingModel
        # Load the embeddings, model information, and script location from a file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        self.action_embeddings = data['action_embeddings']
        self.dialogue_embeddings = data['dialogue_embeddings']
        self.embedding_model = EmbeddingModel(model_name=data['embedding_model_name'])
        self.script_location = data['script_location']
        print(f"Data loaded from {file_path}")
