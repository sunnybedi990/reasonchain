from reasonchain.agent import Agent, MultiAgentSystem
from reasonchain.memory import Memory
from reasonchain.task_utils import assign_and_execute_task, store_in_shared_memory, retrieve_from_shared_memory, collaborate_on_task


# Initialize the Multi-Agent System
multi_agent_system = MultiAgentSystem()

# Create agents with different roles and models
agent1 = Agent(name="AgentAlpha", role="researcher", model_name="gpt-4o", api="openai")
agent2 = Agent(name="AgentBeta", role="strategist", model_name="llama3.1:latest", api="ollama")
agent3 = Agent(name="AgentGamma", role="summarizer", model_name="llama-3.1-8b-instant", api="groq")

# Register agents in the system
multi_agent_system.register_agent(agent1)
multi_agent_system.register_agent(agent2)
multi_agent_system.register_agent(agent3)

# Define a shared task
shared_task = {
    "name": "Improve Remote Work Collaboration",
    "description": "Research best practices, propose strategies, and summarize actionable recommendations.",
    "priority": 4
}

# Add the task to the system
multi_agent_system.add_task(shared_task)

# Step 1: Assign and execute task for AgentAlpha (Researcher)
multi_agent_system.assign_task()
response_alpha = assign_and_execute_task(agent1, "Research best practices for improving remote work collaboration.")

# Step 2: Store research result in shared memory
store_in_shared_memory(agent1.shared_memory, "research", response_alpha)

# Step 3: Collaboration setup for AgentBeta and AgentGamma
task_description = "Strategize and summarize remote work collaboration improvements."
# Collaborate on the task
successful_agents = collaborate_on_task(
    multi_agent_system, 
    ["AgentBeta", "AgentGamma"], 
    task_description
)
# Display successful agents
print("\n=== Successful Collaboration ===")
print(f"Agents involved: {', '.join(successful_agents)}")

# Step 4: AgentBeta analyzes and proposes strategies
research_context = retrieve_from_shared_memory(agent2.shared_memory, "research")
if research_context:
    response_beta = assign_and_execute_task(agent2, f"Propose strategies based on the research: {research_context}")
    store_in_shared_memory(agent2.shared_memory, "strategies", response_beta)
else:
    response_beta = None

# Step 5: AgentGamma summarizes the analysis and strategies
strategies_context = retrieve_from_shared_memory(agent3.shared_memory, "strategies")
if strategies_context:
    response_gamma = assign_and_execute_task(agent3, f"Summarize the strategies into actionable recommendations: {strategies_context}")
else:
    response_gamma = None

# Step 6: Display results
print("\n=== Final Collaboration Results ===")
print(f"AgentAlpha (Researcher): {response_alpha}")
print(f"AgentBeta (Strategist): {response_beta}")
print(f"AgentGamma (Summarizer): {response_gamma}")

