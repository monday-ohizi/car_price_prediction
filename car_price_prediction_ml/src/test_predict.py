from predict import predict_prices

if __name__ == "__main__":
    # Sample cars to test the model
    sample_cars = [
    {
        "Make": "Toyota",
        "Model": "Corolla",
        "Year": 2018,
        "Engine Size": 1.8,
        "Mileage": 45000,
        "Fuel Type": "Petrol",
        "Transmission": "Automatic"
    },
    {
        "Make": "Honda",
        "Model": "Civic",
        "Year": 2017,
        "Engine Size": 2.0,
        "Mileage": 60000,
        "Fuel Type": "Petrol",
        "Transmission": "Manual"
    }
]

# Predict prices
prices = predict_prices(sample_cars)

# Simple, readable output
print("Predicted prices for sample cars:\n")

for i in range(len(sample_cars)):
    make = sample_cars[i]["Make"]
    model = sample_cars[i]["Model"]
    price = prices[i]
    # 2 significant figures
    formatted_price = f"{price:.2f}"
    print(make, model, "→", formatted_price)
    
# Simple summary
print("\nTotal cars predicted:", len(prices))