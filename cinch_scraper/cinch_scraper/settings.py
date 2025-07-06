BOT_NAME = 'cinch_scraper'

SPIDER_MODULES = ['cinch_scraper.spiders']
NEWSPIDER_MODULE = 'cinch_scraper.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performing at the same time to the same domain
CONCURRENT_REQUESTS = 2

# Configure a delay for requests for the same website
DOWNLOAD_DELAY = 3

# Enable or disable downloader middlewares
DOWNLOADER_MIDDLEWARES = {
   'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
   'scrapy_user_agents.middlewares.RandomUserAgentMiddleware': 400,
}

# Configure item pipelines
ITEM_PIPELINES = {
   'scrapy.pipelines.files.FilesPipeline': 1
}

# Enable and configure the AutoThrottle extension (disabled by default)
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 5
AUTOTHROTTLE_MAX_DELAY = 60
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

# Enable showing throttling stats for every response received:
AUTOTHROTTLE_DEBUG = False

# Configure output format
FEED_FORMAT = 'csv'
FEED_URI = 'cinch_cars.csv' 