import pandas as pd
import joblib
import warnings
from features import TitleFeatureExtractor

# Suppress sklearn warnings about feature names lacking
warnings.filterwarnings('ignore')

def main():
    print("==================================================")
    print("   THE RIP-OFF ENGINE : LIVE DEAL CLASSIFIER")
    print("==================================================")
    
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        model = joblib.load('models/best_model.pkl')
    except Exception as e:
        print("Error: Could not load models. Did you finish Milestones 1-3?")
        return

    while True:
        print("\n--- Enter New Listing Details (or type 'quit' to exit) ---")
        title = input("Listing Title: ")
        if title.lower() in ['quit', 'exit', 'q']:
            break
            
        platform = input("Platform (e.g. SNES, N64, PS2): ")
        condition = input("Condition (Mint, Good, Fair, Poor, Untested): ")
        
        try:
            seller_feedback = float(input("Seller Feedback % (0-100): "))
            seller_ratings = int(input("Seller Ratings Count: "))
            listed_price = float(input("Listed Price ($): "))
        except ValueError:
            print("Invalid number entered. Please try again.")
            continue
            
        # Construct DataFrame mapping to what our pipeline expects
        # (It ignores listed_price during transform due to remainder='drop')
        input_data = pd.DataFrame([{
            'title': title,
            'platform': platform,
            'condition': condition,
            'seller_feedback_pct': seller_feedback,
            'seller_ratings_count': seller_ratings,
            'listed_price': listed_price
        }])
        
        try:
            # Predict the fair market value
            X_processed = preprocessor.transform(input_data)
            predicted_fair_value = model.predict(X_processed)[0]
            
            # Calculate classification bounds
            steal_threshold = predicted_fair_value * 0.8
            ripoff_threshold = predicted_fair_value * 1.2
            
            # Determine classification
            if listed_price <= steal_threshold:
                classification = "STEAL! 🔥"
                color_code = '\033[92m' # Green
            elif listed_price >= ripoff_threshold:
                classification = "RIP-OFF! 🛑"
                color_code = '\033[91m' # Red
            else:
                classification = "FAIR DEAL 🤝"
                color_code = '\033[93m' # Yellow
                
            reset_code = '\033[0m'
            
            print("\n--------------------------------------------------")
            print(f"Predicted Fair Value : ${predicted_fair_value:.2f}")
            print(f"Listed Price         : ${listed_price:.2f}")
            print(f"Difference           : ${listed_price - predicted_fair_value:.2f}")
            print(f"Verdict              : {color_code}{classification}{reset_code}")
            print("--------------------------------------------------")
            
        except Exception as e:
            print(f"Error processing input: {e}")

if __name__ == '__main__':
    main()
