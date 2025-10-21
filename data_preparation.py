# ... (previous imports remain the same)

def get_chennai_localities():
    """Return a comprehensive list of unique Chennai localities"""
    return sorted(list(set([
        # Central Chennai
        "Adyar", "Alwarpet", "Anna Nagar", "Besant Nagar", "Chetpet", 
        "Egmore", "George Town", "Kodambakkam", "Mylapore", "Nandanam",
        "Nungambakkam", "Purasawalkam", "Royapettah", "Saidapet", "T. Nagar",
        "Teynampet", "Thiruvanmiyur", "Triplicane", "Chintadripet",
        
        # South Chennai
        "Velachery", "Sholinganallur", "Thoraipakkam", "Perungudi", "Neelankarai",
        "Uthandi", "Kelambakkam", "Siruseri", "Karapakkam", "Taramani", "Guindy",
        "St. Thomas Mount", "Medavakkam", "Pallikaranai", "Perumbakkam",
        "Kottivakkam", "Palavakkam", "Nesapakkam", "Ramapuram",
        
        # West Chennai
        "Porur", "Vadapalani", "K.K. Nagar", "Ashok Nagar", "Virugambakkam",
        "Valasaravakkam", "Mogappair", "Anna Nagar West", "Anna Nagar East",
        "Aminjikarai", "Koyambedu", "Alandur", "Nanganallur", "West Mambalam",
        
        # North Chennai
        "Tondiarpet", "Royapuram", "Washermanpet", "Parrys", "Ayanavaram", 
        "Villivakkam", "Shenoy Nagar", "Kilpauk", "Chetput", "Choolaimedu",
        
        # Suburban Chennai
        "Tambaram", "Chromepet", "Pallavaram", "Keelkattalai", "Perungalathur",
        "Vandalur", "Madipakkam",
        
        # Additional Areas
        "Ambattur", "Avadi", "Poonamallee", "Maduravoyal", "Thirumangalam",
        "Mogappair East", "Mogappair West", "Kolathur", "Vyasarpadi", "Tiruvottiyur",
        "Manali", "Ennore", "Kathivakkam", "Red Hills", "Puzhal", "Madhavaram",
        "Jafferkhanpet", "Kodungaiyur", "Perambur", "KK Nagar", "Thirumangalam",
        "Thirumullaivoyal", "Thiruninravur", "Tiruvallur", "Tiruvotriyur", "Tiruvottiyur",
        "Urapakkam", "Vadapalani", "Valasaravakkam", "Vanagaram", "Vandalur",
        "Velachery", "Villivakkam", "Virugambakkam", "Vyasarpadi", "West Mambalam"
    ])))

def preprocess_data(data_path):
    # Get comprehensive list of unique Chennai localities
    chennai_localities = get_chennai_localities()
    
    # Create a sample dataset with these locations
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'total_sqft': np.random.randint(300, 5000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'location': np.random.choice(chennai_localities, n_samples)
    }
    
    # Generate realistic prices based on area and location
    base_price_per_sqft = np.random.normal(5000, 1000, n_samples)
    location_factors = {loc: np.random.uniform(0.7, 1.5) for loc in chennai_localities}
    
    data['price_lakhs'] = (
        (data['total_sqft'] * base_price_per_sqft * 
         [location_factors[loc] for loc in data['location']]) / 100000
    )
    
    df = pd.DataFrame(data)
    
    # Save processed data
    processed_path = 'data/processed'
    os.makedirs(processed_path, exist_ok=True)
    df.to_csv(os.path.join(processed_path, 'chennai_house_data.csv'), index=False)
    
    return df

# ... (rest of the file remains the same)