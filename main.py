from crewai import Crew
from Agents import agents
from Tasks import tasks

# Define the Crew with agents and tasks
crew = Crew(
    agents= agents,
    tasks=tasks,
    max_rpm=29, # requests per min limit
    verbose=True,
)

input = {
    'industry_name' : 'automotive' 
}

crew.kickoff(inputs=input)
