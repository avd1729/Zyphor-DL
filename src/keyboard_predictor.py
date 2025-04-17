import os
import re
import pickle
import random
from collections import defaultdict, Counter

from ngram_model import NGramModel


class KeyboardPredictor:
    """Main class for keyboard prediction functionality"""
    
    def __init__(self, model_path=None, n=3):
        """
        Initialize the keyboard predictor
        
        Args:
            model_path: Path to load existing model from (or None to create new)
            n: N-gram size for new models
        """
        if model_path and os.path.exists(model_path):
            self.model = NGramModel.load(model_path)
        else:
            self.model = NGramModel(n=n)
        
        self.model_path = model_path or "keyboard_model.pkl"
        self.user_history = []
    
    def add_to_history(self, text):
        """Add user input to history for later training"""
        self.user_history.append(text)
        
        # If history gets too long, train on it and clear
        if len(self.user_history) >= 100:
            self.train_on_history()
    
    def train_on_history(self):
        """Train the model on collected user history"""
        if not self.user_history:
            return
            
        # Combine all history entries and train
        all_text = " ".join(self.user_history)
        self.model.train(all_text)
        
        # Save updated model
        self.save_model()
        
        # Clear history after training
        self.user_history = []
    
    def predict(self, context, num_predictions=3):
        """Predict next words based on context"""
        return self.model.predict_next_word(context, num_predictions)
    
    def save_model(self):
        """Save the model to the specified path"""
        self.model.save(self.model_path)
    
    def export_for_aggregation(self, export_path=None):
        """
        Export model for server aggregation
        This could be enhanced with additional metadata in a real implementation
        """
        if not export_path:
            export_path = "keyboard_model_export.pkl"
        self.model.save(export_path)
        return export_path


# Example usage
def demo():
    # Sample training data
    sample_texts = [
        "I would like to thank you for your support.",
        "I would like to see you tomorrow.",
        "Would you like to go to the movies?",
        "I will be there in a minute.",
        "I will call you back as soon as possible.",
        "Can you please send me the document?",
        "Please let me know if you need anything else."
    ]
    
    # Create a predictor
    predictor = KeyboardPredictor(n=3)
    
    # Train on sample data
    for text in sample_texts:
        predictor.add_to_history(text)
    
    predictor.train_on_history()
    
    # Test predictions
    test_contexts = [
        "I would like to",
        "I will",
        "Please let me",
        "Can you please"
    ]
    
    print("=== Sample Predictions ===")
    for context in test_contexts:
        predictions = predictor.predict(context)
        print(f"Context: '{context}'")
        print(f"Predictions: {predictions}")
        print()
    
    # Export model for aggregation
    export_path = predictor.export_for_aggregation()
    print(f"Model exported for aggregation to: {export_path}")


if __name__ == "__main__":
    demo()