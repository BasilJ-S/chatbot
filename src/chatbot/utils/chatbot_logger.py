import logging

# Configure logging once at the top of your script
logging.basicConfig(
    filename="logs/chatbot.log",  # or omit to log to console
    level=logging.INFO,  # set level
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)
logger = logging.getLogger(__name__)
