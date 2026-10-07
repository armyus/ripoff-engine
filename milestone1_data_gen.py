import pandas as pd
import numpy as np
import os
import random

def generate_synthetic_data(num_samples=3000, output_path='data/dataset.csv'):
    np.random.seed(42)
    random.seed(42)
    
    platforms = ['NES', 'SNES', 'N64', 'GameCube', 'PS1', 'PS2', 'Sega Genesis']
    base_prices = {
        'NES': 40, 'SNES': 60, 'N64': 50, 'GameCube': 80, 
        'PS1': 35, 'PS2': 45, 'Sega Genesis': 30
    }
    
    conditions = ['Mint', 'Good', 'Fair', 'Poor', 'Untested']
    condition_multipliers = {
        'Mint': 1.5, 'Good': 1.0, 'Fair': 0.7, 'Poor': 0.4, 'Untested': 0.3
    }
    
    # Generate titles
    adjectives = ['Rare', 'Vintage', 'Bundle', 'Tested', 'Working', 'CIB', 'Boxed', 'Loose', 'Scratch', 'Flaw']
    games = ['Mario', 'Zelda', 'Pokemon', 'Final Fantasy', 'Sonic', 'Metroid', 'Castlevania', 'Mega Man', 'Donkey Kong']
    
    data = []
    
    for _ in range(num_samples):
        platform = random.choice(platforms)
        condition = random.choice(conditions)
        
        # Base fair value based on platform and condition
        fair_value = base_prices[platform] * condition_multipliers[condition]
        
        # Add some random noise to fair value to simulate different games
        game_multiplier = np.random.lognormal(mean=0.2, sigma=0.6)
        fair_value *= game_multiplier
        
        # Generate messy title
        title_parts = []
        if random.random() > 0.5: title_parts.append(random.choice(adjectives))
        title_parts.append(random.choice(games))
        title_parts.append(platform)
        if random.random() > 0.5: title_parts.append(random.choice(adjectives))
        
        # Randomize capitalization and add some symbols to make it messy
        title = " ".join(title_parts)
        if random.random() > 0.7:
            title = title.upper()
        elif random.random() > 0.8:
            title = title.lower()
            
        if random.random() > 0.8:
            title += " !!!"
        elif random.random() > 0.8:
            title += " L@@K"
            
        # Simulate listed price based on fair value + noise (some people overprice, some underprice)
        # 15% steal, 65% fair, 20% ripoff
        listing_type = np.random.choice(['STEAL', 'FAIR', 'RIP-OFF'], p=[0.15, 0.65, 0.20])
        
        if listing_type == 'STEAL':
            # At least 20% below fair value (multiplier < 0.8)
            listed_price = fair_value * np.random.uniform(0.4, 0.79)
        elif listing_type == 'FAIR':
            # Within +/- 20% (multiplier between 0.8 and 1.2)
            listed_price = fair_value * np.random.uniform(0.8, 1.2)
        else: # RIP-OFF
            # At least 20% above fair value (multiplier > 1.2)
            listed_price = fair_value * np.random.uniform(1.21, 3.0)
            
        # Seller feedback (0 to 100%) - left skewed
        seller_feedback = np.random.beta(a=8, b=2) * 100
        
        # Number of ratings - log normal
        seller_ratings = int(np.random.lognormal(mean=4, sigma=1.5))
        
        data.append({
            'title': title,
            'platform': platform,
            'condition': condition,
            'seller_feedback_pct': round(seller_feedback, 2),
            'seller_ratings_count': seller_ratings,
            'listed_price': round(listed_price, 2),
            'true_fair_value': round(fair_value, 2)
        })
        
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_samples} records and saved to {output_path}")
    
    # Quick statistics
    print("\nDataset Statistics:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Average Listed Price: ${df['listed_price'].mean():.2f}")
    
if __name__ == '__main__':
    generate_synthetic_data()
