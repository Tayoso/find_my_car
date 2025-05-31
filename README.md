# Find My Car Assistant

An intelligent car recommendation system that helps users find their ideal car based on natural language requirements. The application uses the Hugging Face transformers library for natural language understanding and provides personalized car recommendations.

## Features

- Natural language understanding of car requirements
- Intelligent filtering based on multiple criteria:
  - Family-friendly vehicles (SUVs, wagons)
  - Long-distance capability (hybrid/diesel vehicles)
  - Durability (age and mileage considerations)
  - Fuel efficiency
  - Luxury options
  - Budget considerations
  - Vehicle size preferences
- Clean and intuitive chat interface
- CSV-based car database support
- Real-time recommendations

## Requirements

- Python 3.9+
- Hugging Face account and API token
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/find_my_car.git
cd find_my_car
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Hugging Face token:
```
HF_TOKEN=your_token_here
```

## Usage

1. Prepare your car database CSV file with the following columns:
   - make
   - model
   - age
   - body_type
   - fuel_type
   - transmission_type
   - mileage
   - cost

2. Run the application:
```bash
streamlit run app.py
```

3. Upload your car database CSV file through the web interface

4. Start chatting with the assistant about your car requirements!

## Example Queries

- "I want a family car that can go long distance and is very durable"
- "Looking for a fuel-efficient car for city driving"
- "Need a luxury SUV with good mileage"
- "Want a budget-friendly compact car"

## Project Structure

- `app.py`: Main Streamlit application
- `test_output.py`: Test script for the recommendation system
- `requirements.txt`: Python package dependencies
- `.env`: Environment variables (not tracked in git)
- `sample_cars.csv`: Example car database

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 