import pandas as pd
import numpy as np
import os

def generate_realistic_addis_houses(n=1500):
    np.random.seed(42)
    
    # Neighborhoods with their specific price multipliers (Base price per m2 in ETB)
    neighborhoods = {
        'Bole': 180000,
        'Old Airport': 175000,
        'Kazanchis': 170000,
        'Sarbet': 160000,
        'CMC': 120000,
        'Lebu': 110000,
        'Ayat': 100000,
        'Summit': 95000,
        'Jemo': 85000,
        'Gullele': 80000,
        'Akaki': 65000,
        'Piassa': 150000
    }
    
    house_types = ['Villa', 'Apartment', 'Condominium']
    
    data = []
    for _ in range(n):
        loc = np.random.choice(list(neighborhoods.keys()))
        h_type = np.random.choice(house_types, p=[0.25, 0.40, 0.35])
        
        # Base Area based on type - reflecting typical Addis sizes
        if h_type == 'Villa':
            area = np.random.randint(150, 600)
            rooms = np.random.randint(4, 10)
            baths = np.random.randint(3, 7)
        elif h_type == 'Apartment':
            area = np.random.randint(80, 220)
            rooms = np.random.randint(2, 5)
            baths = np.random.randint(1, 4)
        else: # Condominium
            area = np.random.randint(35, 110)
            rooms = np.random.randint(1, 3)
            baths = np.random.randint(1, 2)
            
        age = np.random.randint(0, 35)
        
        # Distance from city center logic (0 to 18km)
        dist_map = {
            'Piassa': 0, 'Kazanchis': 1, 'Bole': 4, 'Sarbet': 3,
            'Old Airport': 5, 'CMC': 10, 'Ayat': 13, 'Summit': 12,
            'Lebu': 9, 'Jemo': 14, 'Gullele': 6, 'Akaki': 18
        }
        dist = dist_map[loc]
        
        # Price Calculation logic
        # Multipliers based on User feedback: Condo (cheapest) < Apartment < Villa (expensive)
        type_mult = {
            'Condominium': 0.5,   # Basic affordable housing
            'Apartment': 2.2,     # Modern high-rise living
            'Villa': 7.0          # Luxury private estates
        }
        
        base_price = (area * neighborhoods[loc]) * type_mult[h_type]
        
        # Adding value for rooms/baths and deducting for age/distance
        price = base_price + \
                (rooms * 600000) + \
                (baths * 400000) - \
                (age * 200000) - \
                (dist * 300000)
        
        # Noise for realism
        noise = np.random.normal(0, base_price * 0.08)
        final_price = max(1800000, price + noise) # Minimum price floor in ETB
        
        data.append([loc, h_type, area, rooms, baths, age, dist, round(final_price, -3)])
        
    df = pd.DataFrame(data, columns=['Location', 'Type', 'Area', 'Bedrooms', 'Bathrooms', 'Age', 'Distance_to_Center', 'Price'])
    
    output_path = 'd:/Project/Ml/ethio_ml_hub/data/addis_housing.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated {n} records with refined hierarchy (Condo < Apartment < Villa)")

if __name__ == "__main__":
    generate_realistic_addis_houses()
