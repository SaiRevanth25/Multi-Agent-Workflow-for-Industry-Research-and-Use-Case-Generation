from dotenv import load_dotenv
load_dotenv()
import os
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')
search = SerperDevTool()
scrape = ScrapeWebsiteTool()

tools = [search, scrape]