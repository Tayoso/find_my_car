# Cinch Car Scraper

This is a web scraper built with Scrapy to extract car listings from Cinch's website.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the scraper:
```bash
scrapy crawl cinch
```

The scraper will:
- Extract car details from Cinch's website
- Follow pagination to get all available listings
- Save results to `cinch_cars.csv`

## Output Format

The scraper extracts the following information for each car:
- Make
- Model
- Year
- Mileage
- Fuel type
- Transmission
- Full price
- Monthly price
- Scrape date
- URL

## Notes

- The scraper respects robots.txt and includes delays between requests
- It uses a rotating user agent to avoid being blocked
- Results are saved in CSV format for easy import into the car recommendation system 