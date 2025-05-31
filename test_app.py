import pandas as pd
from transformers import pipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test data
test_data = '''make,model,age,body_type,fuel_type,transmission_type,mileage,cost
Toyota,RAV4,2,suv,hybrid,automatic,25000,32000
Honda,CR-V,3,suv,petrol,automatic,35000,28000
Volkswagen,Passat,4,wagon,diesel,automatic,45000,26000
Toyota,Camry,1,sedan,hybrid,automatic,15000,30000
Ford,Explorer,2,suv,petrol,automatic,28000,35000'''

# Save test data
with open('test_cars.csv', 'w') as f:
    f.write(test_data)

# Load test data
df = pd.read_csv('test_cars.csv')

# Initialize classifier
print("Initializing classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    token=os.getenv("HF_TOKEN")
)

# Test query
test_query = "I want a family car that can go long distance and very durable"

# Define categories
categories = [
    "family car", "long distance", "durable", "fuel efficient",
    "luxury", "sporty", "budget friendly", "compact"
]

print("\nTesting classification...")
try:
    # Single classification for all categories
    result = classifier(
        inputs=test_query,
        candidate_labels=categories,
        multi_label=True
    )
    
    print("\nRaw result:", result)
    
    # Extract categories with high confidence
    selected_categories = []
    for label, score in zip(result['labels'], result['scores']):
        print(f"\n{label}: {score}")
        if score > 0.7:
            selected_categories.append(label)
            
    print("\nSelected categories:", selected_categories)
    
except Exception as e:
    print(f"Error during classification: {str(e)}")

# Test filtering
if selected_categories:
    print("\nFiltering cars...")
    filtered_df = df.copy()
    
    if "family car" in selected_categories:
        filtered_df = filtered_df[filtered_df["body_type"].isin(["suv", "wagon"])]
    if "long distance" in selected_categories:
        filtered_df = filtered_df[filtered_df["fuel_type"].isin(["hybrid", "diesel"])]
    if "durable" in selected_categories:
        filtered_df = filtered_df[
            (filtered_df["age"] <= 5) & 
            (filtered_df["mileage"] <= 80000)
        ]
    
    print("\nFiltered cars:")
    print(filtered_df.to_string()) 