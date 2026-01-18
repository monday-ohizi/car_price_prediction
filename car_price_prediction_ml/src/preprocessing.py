from difflib import get_close_matches
import pandas as pd
import numpy as np

# Allowed categories per column (from EDA)
allowed_categories = {
    "Make": ["Toyota", "Ford", "Honda", "Mercedes", "BMW"],
    "Model": ["Civic", "Corolla", "Focus", "C-Class", "360I"],
    "Fuel Type": ["Hybrid", "Diesel", "Petrol", "Electric"],
    "Transmission": ["Automatic", "Manual"]
}

MATCH_THRESHOLD = 0.8  # minimum similarity ratio to accept a match

def clean_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    for col in df.select_dtypes(include='object').columns:
        # Step 1: Basic cleaning
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"[^a-zA-Z\s]", "", regex=True)
            .str.title()
            .replace("", pd.NA)
        )
        
        # Step 2: Domain corrections (from EDA)
        if col == "Model":
            df[col] = df[col].replace({"Cclass": "C-Class", "I": "360I"})
        if col == "Make":
            df[col] = df[col].replace({"Bmw": "BMW"})
        
        # Step 3: Map unknown values to allowed categories using fuzzy matching
        def map_value(val):
            if pd.isna(val):
                return val  # leave NaN for pipeline imputer
            match = get_close_matches(val, allowed_categories[col], n=1, cutoff=MATCH_THRESHOLD)
            return match[0] if match else "Other"
        
        df[col] = df[col].apply(map_value)
        
    return df
	
	
# Numerical column cleaning as applied in notebook 1
def clean_numerical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include= ['int64','float64']).columns.drop("Price",errors="ignore"):
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    if "Year" in df.columns:
            current_year = pd.Timestamp.now().year
            df.loc[~df['Year'].between(1925, current_year), 'Year'] = pd.NA
            
    if "Mileage" in df.columns:
            df.loc[(df["Mileage"] < 0) | (df["Mileage"] > 600000), "Mileage"] = pd.NA
            
    if "Engine Size" in df.columns:
            df.loc[(df["Engine Size"] < 0) | (df["Engine Size"] > 6), "Engine Size"] =pd.NA
            
    return df
	
	
	# Feature engineering as proved in notebook 1
def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Year" in df.columns:
        current_year = pd.Timestamp.now().year
        df["Car Age"] = current_year - df["Year"]
        df.drop(columns= 'Year', inplace=True)
        
    if "Mileage" in df.columns:
        df["Log Mileage"] = np.log1p(df["Mileage"])
        df.drop(columns= "Mileage", inplace=True)
    
    return df