import pickle

# Path to the downloaded aggregated model
aggregated_model_path = 'models/aggregated_model.pkl'

def test_aggregated_model(model_path):
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Check for essential keys
        required_keys = ['n', 'vocabulary', 'total_words']
        if not all(key in model_data for key in required_keys):
            print(f"Model missing required keys. Missing keys: {[key for key in required_keys if key not in model_data]}")
            return

        # Print some basic information about the model
        print(f"Model loaded successfully!")
        print(f"Vocabulary size: {len(model_data['vocabulary'])}")
        print(f"Total words count: {model_data['total_words']}")
        
        vocabulary = model_data['vocabulary']
        print(f"Vocabulary is a set, here are some example words: {list(vocabulary)[:10]}")  # Show first 10 words from the set
        
        print(f"Model n value (example state): {model_data['n']}")

    except Exception as e:
        print(f"Error loading or processing the model: {e}")

if __name__ == "__main__":
    test_aggregated_model(aggregated_model_path)
