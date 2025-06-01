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
- uv package manager (faster Python package installation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tayoso/find_my_car.git
cd find_my_car
```

2. Install uv (if not already installed):
```bash
pip install uv
```

3. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate
```

4. Install the package in development mode:
```bash
uv pip install -e .
```

5. Create a `.env` file in the project root and add your Hugging Face token:
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

   A sample database is provided in `find_my_car/data/sample_cars.csv`.

2. Run the application:
```bash
streamlit run app.py
```

3. Upload your car database CSV file through the web interface

4. Start chatting with the assistant about your car requirements!

## Example Queries

- "I want a family car that can go long distance and very durable."
- "What's the best SUV under $30,000?"
- "I need a fuel-efficient car with low mileage"
- "Show me automatic transmission cars with less than 30,000 miles"
- "What's the newest electric car in the database?"

## Project Structure

```
find_my_car/
├── __init__.py           # Package initialization
├── app.py               # Streamlit web interface
├── recommender.py       # Core recommendation logic
└── data/
    └── sample_cars.csv  # Example car database
```

## Development

This project uses modern Python tooling:
- `pyproject.toml` for project configuration
- `uv` for fast, reliable package management
- `ruff` for linting and formatting

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 