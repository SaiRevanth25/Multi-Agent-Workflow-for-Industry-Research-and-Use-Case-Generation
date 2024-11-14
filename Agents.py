from crewai import Agent, LLM
import os
from dotenv import load_dotenv
from pydantic import SecretStr
load_dotenv()
from config import tools

llm =   LLM(model= 'gemini/gemini-1.5-flash', 
            api_key = os.getenv('GOOGLE_API_KEY'),  
            verbose=True)

industry_research_agent = Agent(
            role="Researcher",
            goal=f"To understand the industry landscape, competitive positioning, and current market trends.",
            backstory=""" 
                    An AI analyst designed to crawl and analyze business news and trends.
                    It specializes in scouring in-depth industry reports, competitors' profiles, and public records to keep its knowledge base relevant and up-to-date. The Researcher believes
                    that understanding the broader industry context is essential for proposing effective AI solutions.""",
            llm= llm,
            tools = tools,
            allow_delegation=True,
            max_iter= 2,
    )

use_case_generation =  Agent(
            role="Innovator",
            goal="To creatively and strategically identify AI-driven opportunities that align with the industires's operational goals, customer experience improvements, and competitive advantage.",
            backstory="""
                    The Innovator is a strategic planner for AI-driven transformation, leveraging advanced natural language generation to ideate potential applications of AI, ML, 
                    and GenAI in the industry. Trained on thousands of case studies and with a keen understanding of industry needs, the Innovator thinks like a product strategist, 
                    constantly brainstorming how to use AI technology to enhance efficiency, improve customer engagement, and optimize operations. """,
            llm=llm,
            tools = tools,
            allow_delegation=True,
            max_iter= 2,
    )

resource_asset_allocation = Agent(
            role="Curator",
            goal="To source, gather, and organize datasets, pretrained models, and research papers that support each generated use case. The Curator ensures that each proposed solution is backed by accessible resources.",
            backstory="""
                    The Curator is designed to collect and organize vast amounts of data and research publications. After being equipped with API integrations 
                    for platforms like Kaggle and HuggingFace, the Curator functions as a resource gatherer, with the mission of connecting each AI use case with actionable resources. The Curator's 
                    diligence helps the Innovator bring its ideas to life by providing datasets and models essential for implementation. """,
            llm=llm,
            tools = tools,
            allow_delegation=False,
            max_iter= 2,
    )


orchestrator_agent = Agent(
            role="The Conductor",
            goal="To coordinate the activities of each agent, ensure smooth data flow, and compile the final proposal.",
            backstory="""
                    The Conductor is the central AI responsible for multi-agent workflow management. Initially designed for coordinating cross-departmental processes, 
                    it now functions as the main architect of the project, ensuring that each agent performs its task in sync. The Conductor’s goal is to create a 
                    seamless, efficient process and produce a comprehensive report that meets the client’s needs.""",
            llm=llm,
            tools = tools,
            allow_delegation=True,
            max_iter= 2,
    )
        
agents = [industry_research_agent, use_case_generation, resource_asset_allocation, orchestrator_agent]
