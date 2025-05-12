from joblib import load
import pandas as pd
from pymongo import MongoClient
import numpy as np
from llama_cpp import Llama
import re

# Initialize the model
llm = Llama.from_pretrained(
    repo_id="DavidAU/Gemma-The-Writer-Mighty-Sword-9B-GGUF",
    filename="Gemma-The-Writer-Mighty-Sword-9B-D_AU-IQ4_XS.gguf",
)

# Load the Skyrim Weapons CSV
sw = pd.read_csv('Skyrim_Weapons.csv')

# Load the trained model and encoder
model = load("model.pkl")
encoder = load("ordinal_encoder.pkl")

# Categorical columns used in training
categorical_cols = ['Name', 'Upgrade', 'Perk', 'Type', 'Category']

# Helper function to generate a weapon's details
def generate_weapon(name):
    # Update the prompt to ensure the AI provides a single number for Damage and Weight
    prompt = f"""
        Generate a Skyrim weapon named '{name}' with the following attributes:
        - Damage: A single integer value (e.g., 15).
        - Weight: A single integer value (e.g., 10).
        - Upgrade: The upgrade material (e.g., Diamond Ingot, Steel to Daedric).
        - Perk: A unique perk (e.g., Frostbite Cleave).
        - Type: The type of weapon (e.g., Sword, Axe, Bow).
        - Category: The category of the weapon (e.g., Melee, Ranged).

        Do not use ranges for damage (e.g., 18-25), instead provide a single integer value. Format the output as:
        Damage: <value>, Weight: <value>, Upgrade: <value>, Perk: <value>, Type: <value>, Category: <value>.

        Ensure that Damage is an integer and Weight is a decimal number. 
        """

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7
    )

    generated_text = response["choices"][0]["message"]["content"]
    weapon_details = parse_generated_text(generated_text)
    return weapon_details

import re

def parse_generated_text(text):
    weapon_data = {}

    # Remove any markdown symbols (like **, etc.)
    text = re.sub(r"\*\*|\#", "", text)

    # Normalize by replacing newlines and ensuring commas are clean
    text = text.replace("\n", " ").replace("  ", " ")

    # Extract key-value pairs
    pattern = r"(Damage|Weight|Upgrade|Perk|Type|Category):\s*([^:,]+(?: [^:,]+)*)"
    matches = re.findall(pattern, text)

    for key, value in matches:
        key = key.strip()
        value = value.strip().strip(".")

        if key == "Damage":
            try:
                if '-' in value:
                    low, high = map(int, value.split('-'))
                    weapon_data[key] = (low + high) // 2
                else:
                    weapon_data[key] = int(value)
            except ValueError:
                print(f"❌ Invalid Damage format: '{value}'")
                return {}
        elif key == "Weight":
            try:
                weapon_data[key] = float(value)
            except ValueError:
                print(f"❌ Invalid Weight format: '{value}'")
                return {}
        else:
            weapon_data[key] = value

    # Final validation
    required_fields = ["Damage", "Weight", "Upgrade", "Perk", "Type", "Category"]
    missing = [f for f in required_fields if f not in weapon_data]
    if missing:
        print(f"❌ Not all required fields found. Missing: {missing}")
        print(f"Parsed fields: {weapon_data}")
        return {}

    print("✅ Parsed fields:", weapon_data)
    return weapon_data

# Parse Damage, in case it's a range (e.g., '15-22') or a single value
def parse_damage(damage):
    # If damage is a range, take the average of the range
    if '-' in damage:
        try:
            damage_range = damage.split('-')
            return (float(damage_range[0].strip()) + float(damage_range[1].strip())) / 2
        except ValueError:
            return None
    # If it's a single number, just convert it
    try:
        return float(damage)
    except ValueError:
        return None

# Parse Weight
def parse_weight(weight):
    try:
        return float(weight)
    except ValueError:
        return None

# Get the new weapon's name and generate the rest of its features
weapon_name = "Cooler axe"  # Change this to the name you want
new_weapon = generate_weapon(weapon_name)

# If new weapon is valid, proceed with further steps
if new_weapon:
    # Convert new_weapon to DataFrame
    new_df = pd.DataFrame([new_weapon])

    # Combine with the training dataset to ensure column consistency
    combined = pd.concat([new_df, sw], ignore_index=True)

    # One-hot encode the categorical variables (same columns as during training)
    combined_encoded = pd.get_dummies(combined, drop_first=True)

    # Ensure the new data has the same columns as the training set
    training_columns = ['Damage', 'Weight', 'Upgrade', 'Perk', 'Type', 'Category']
    combined_encoded = combined_encoded.reindex(columns=training_columns, fill_value=0)

    # Take the first row (which is the new weapon)
    new_encoded = combined_encoded.iloc[[0]]

    # Predict using the trained model
    predicted_price = model.predict(new_encoded)[0]

    # Save to MongoDB
    new_weapon["predicted_price"] = round(predicted_price, 2)

    # Fixing MongoDB Insert Issue
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["Skyrim_weapons"]
        collection = db["weapons"]
        collection.insert_one(new_weapon)
        print(f"Weapon added to MongoDB: {new_weapon}")
    except Exception as e:
        print(f"Failed to insert into MongoDB: {e}")

    # Output the result
    print(f"Predicted price for '{new_weapon['Type']}': {predicted_price}")
else:
    print("Failed to generate valid weapon.")
