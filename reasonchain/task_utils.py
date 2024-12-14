def dynamic_complexity_evaluator(step):
    """
    Dynamically evaluate the complexity of a step.
    :param step: Step description.
    :return: Complexity level ('low', 'medium', 'high').
    """
    if "fetch" in step.lower() or "retrieve" in step.lower():
        return "medium"
    elif "generate" in step.lower() or "evaluate" in step.lower():
        return "high"
    else:
        return "low"


def assign_and_execute_task(agent, task_description):
    """
    Assign a task to an agent, execute it using a CoT pipeline, and return the result.
    """
    from reasonchain.cot_pipeline import CoTPipeline

    cot_pipeline = CoTPipeline(agent=agent)
    cot_pipeline.add_step(task_description)
    return agent.reason(cot_pipeline)


def store_in_shared_memory(memory, key, value):
    """
    Store a result in shared memory with a specific key.
    """
    if value:
        memory.add_entry(key, value)


def retrieve_from_shared_memory(memory, key):
    """
    Retrieve a value from shared memory by key.
    """
    return memory.retrieve_entry(key)


def collaborate_on_task(multi_agent_system, agent_names, task_description):
    """
    Enable multiple agents to collaborate on a shared task.
    :return: List of agents successfully involved in collaboration.
    """
    successful_agents = []
    for agent_name in agent_names:
        if agent_name in multi_agent_system.agents:
            multi_agent_system.agents[agent_name].observe({"task": task_description})
            successful_agents.append(agent_name)
        else:
            print(f"Agent {agent_name} is not registered in the system.")
    return successful_agents


