import requests
import re
from pymongo import MongoClient
import pandas as pd
from joblib import load

# Load the Skyrim Weapons CSV
sw = pd.read_csv('Skyrim_Weapons.csv')

# Load the trained model and encoder
model = load("model.pkl")
encoder = load("ordinal_encoder.pkl")

# Categorical columns used in training
categorical_cols = ['Name', 'Upgrade', 'Perk', 'Type', 'Category']

API_URL = "http://localhost:11434/api/generate"


# ⭐ Function to generate a full weapon name from a base name
def generate_weapon_name(base_name):
    prompt = f"""
        Create a Skyrim-style weapon name using the base '{base_name}'.
        The name should sound mystical or powerful, and follow formats like:
        - {base_name}'s Icefang Blade
        - {base_name}'s Vengeance
        - Blade of {base_name}

        Return only the name, no description.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"model": "hf.co/DavidAU/Gemma-The-Writer-Mighty-Sword-9B-GGUF:Q2_K", "prompt": prompt, "stream": False}

    try:
        print("Generating weapon name...")
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        name_text = response.json().get("response", "").strip()
        return name_text
    except Exception as e:
        print(f"❌ Error generating weapon name: {e}")
        return f"{base_name}'s Weapon"


# Function to generate the full weapon stats
def generate_weapon(full_name):
    prompt = f"""
        Generate a Skyrim weapon named '{full_name}' with the following attributes:
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
    headers = {"Content-Type": "application/json"}
    payload = {"model": "hf.co/DavidAU/Gemma-The-Writer-Mighty-Sword-9B-GGUF:Q2_K", "prompt": prompt, "stream": False}

    try:
        print("Sending request to AI for weapon stats...")
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        generated_text = response.json().get("response", "")
        print("Generated text from AI:", generated_text)

        weapon_details = parse_generated_text(generated_text)
        weapon_details["Name"] = full_name  # ⭐ Add the generated name
        return weapon_details
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Dockerized AI API: {e}")
        return {}


# Function to parse the generated text
def parse_generated_text(text):
    print("TEXT INPUT TO PARSER:")
    print(repr(text))

    weapon_data = {}
    pattern = r"([A-Za-z]+):\s*([^,]+)"  # Match the pattern "Attribute: Value"

    lines = text.strip().splitlines()
    for line in lines:
        matches = re.findall(pattern, line)  # Find all matches for the pattern
        for key, value in matches:
            if key == "Damage":
                weapon_data[key] = int(value)  # Convert Damage to int
            elif key == "Weight":
                weapon_data[key] = float(value)  # Convert Weight to float
            else:
                weapon_data[key] = value.strip()  # Keep the rest as string (e.g., Upgrade, Perk, etc.)

    print("PARSED FIELDS:", weapon_data)

    required_fields = ["Damage", "Weight", "Upgrade", "Perk", "Type", "Category"]
    missing = [f for f in required_fields if f not in weapon_data]

    if missing:
        print(f"❌ Not all required fields found. Missing: {missing}")
        return {}

    print("✅ Parsed weapon:", weapon_data)
    return weapon_data


# ⭐ Ask for input name and generate full weapon name
base_input = input("Enter a base name for the weapon (e.g., Hilda): ").strip()
generated_name = generate_weapon_name(base_input)
print(f"Generated weapon name: {generated_name}")

# Generate the rest of the weapon details
new_weapon = generate_weapon(generated_name)

# If valid, predict price and insert into MongoDB
if new_weapon:
    new_df = pd.DataFrame([new_weapon])
    combined = pd.concat([new_df, sw], ignore_index=True)
    combined_encoded = pd.get_dummies(combined, drop_first=True)

    training_columns = ['Damage', 'Weight', 'Upgrade', 'Perk', 'Type', 'Category']
    combined_encoded = combined_encoded.reindex(columns=training_columns, fill_value=0)
    new_encoded = combined_encoded.iloc[[0]]

    predicted_price = model.predict(new_encoded)[0]
    new_weapon["predicted_price"] = round(predicted_price, 2)

    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["Skyrim_weapons"]
        collection = db["weapons"]
        collection.insert_one(new_weapon)
        print(f"✅ Weapon added to MongoDB: {new_weapon}")
    except Exception as e:
        print(f"❌ Failed to insert into MongoDB: {e}")

    print(f"Predicted price for '{new_weapon['Name']}': {predicted_price}")
else:
    print("❌ Failed to generate valid weapon.")
