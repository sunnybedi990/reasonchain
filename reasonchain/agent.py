from .memory import Memory, SharedMemory
from .llm_models.model_manager import ModelManager
import logging
import time
import heapq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class Agent:
    def __init__(self, name, model_name="gpt-4", api='openai', memory=None, shared_memory=None, role="generalist"):
        """
        Initialize the agent.
        :param name: Name of the agent.
        :param model: LLM model instance.
        :param memory: Memory instance for storing context.
        """
        self.name = name
        self.role = role
        self.model_manager = ModelManager(model_name=model_name, api=api)
        self.memory = memory or Memory()  # Use Memory class
        self.shared_memory = shared_memory  # Assigned during registration
        self.is_busy = False  # Agent availability status
        self.completed_tasks = []


    def observe(self, input_data):
        """
        Process input data.
        :param input_data: Raw input for the agent.
        :return: Processed data.
        """
        try:
            if not isinstance(input_data, (str, dict)):
                raise ValueError("Input data must be a string or dictionary.")
            self.memory.store_short_term(input_data)
            logging.info(f"[{self.name}] Observing input: {input_data}")
            return input_data
        except Exception as e:
            logging.error(f"[{self.name}] Error observing input: {e}")
            return None

    def reason(self, cot_pipeline, use_rag=False, rag_query=None, top_k=5):
        """
        Execute the CoT pipeline for reasoning, optionally using RAG for context.
        :param cot_pipeline: CoTPipeline instance.
        :param use_rag: Whether to augment reasoning with RAG.
        :param rag_query: Query string for RAG (if applicable).
        :param top_k: Number of RAG results to retrieve.
        :return: Reasoned response.
        """
        try:
            logging.info(f"[{self.name}] Starting reasoning process...")

            # Retrieve short-term context
            context = self.memory.retrieve_short_term()
            context_str = " | ".join(
                [f"{key}: {value}" for entry in context for key, value in entry.items()]
                if context and isinstance(context[0], dict) else context
            ) or "No context available."

            # Optionally retrieve RAG context
            rag_context = []
            if use_rag and rag_query:
                rag_context = self.retrieve_rag_context(rag_query, top_k=top_k)

            # Combine contexts
            combined_context = f"Short-term context: {context_str}\nRAG context: {rag_context}" if rag_context else context_str
            logging.info(f"[{self.name}] Combined context for reasoning: {combined_context}")

            # Execute CoT pipeline
            if not hasattr(cot_pipeline, "execute"):
                raise AttributeError("Provided CoT pipeline is invalid or missing an 'execute' method.")
            response = cot_pipeline.execute(self.model_manager, combined_context)
            return response
        except Exception as e:
            logging.error(f"[{self.name}] Error during reasoning: {e}")
            return None


        
    def act(self, response):
        """
        Act based on the response.
        :param response: Reasoned output.
        """
        try:
            if response:
                self.memory.store_long_term(response)
                logging.info(f"[{self.name}] Acting on response: {response}")
                self.completed_tasks.append(response)
                return response
            else:
                logging.warning(f"[{self.name}] No response to act upon.")
                return None
        except Exception as e:
            logging.error(f"[{self.name}] Error during act: {e}")
            return None
    
    def receive_messages(self, messages):
        """
        Process received messages.
        :param messages: List of messages.
        """
        try:
            for message in messages:
                logging.info(f"[{self.name}] Received message from {message['from']}: {message['content']}")
        except Exception as e:
            logging.error(f"[{self.name}] Error receiving messages: {e}")

    def switch_task(self, new_task):
        """
        Switch to a new task and clear the context.
        :param new_task: The new task to switch to.
        """
        try:
            logging.info(f"[{self.name}] Switching to new task: {new_task}")
            self.memory.clear_short_term()
            self.observe(new_task)
        except Exception as e:
            logging.error(f"[{self.name}] Error switching task: {e}")

    def monitor_health(self):
        """
        Monitor the health and status of the agent.
        :return: A dictionary with health status details.
        """
        try:
            health_status = {
                "memory_usage": len(self.memory.retrieve_short_term()),
                "is_busy": self.is_busy,
                "completed_tasks": len(self.completed_tasks)
            }
            logging.info(f"[{self.name}] Health status: {health_status}")
            return health_status
        except Exception as e:
            logging.error(f"[{self.name}] Error monitoring health: {e}")
            return None

    def prioritize_task(self, tasks):
        """
        Prioritize tasks based on their priority field.
        :param tasks: List of tasks.
        :return: Sorted list of tasks.
        """
        try:
            if not all(isinstance(task, dict) and "priority" in task for task in tasks):
                raise ValueError("Some tasks are missing a 'priority' field.")
            return sorted(tasks, key=lambda x: x.get("priority", 0), reverse=True)
        except Exception as e:
            logging.error(f"[{self.name}] Error prioritizing tasks: {e}")
            return tasks

    def perform_task_with_timeout(self, task, timeout=30):
        """
        Perform a task within a specified timeout period.
        :param task: The task to perform.
        :param timeout: Timeout duration in seconds.
        :return: The result of the task or a timeout warning.
        """
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                result = self.observe(task)
                if result:
                    return result
            logging.warning(f"[{self.name}] Task timeout exceeded.")
            return None
        except Exception as e:
            logging.error(f"[{self.name}] Error during task execution with timeout: {e}")
            return None

    def track_task_progress(self, task_id, status):
        """
        Update and log the progress of a task.
        :param task_id: Unique identifier for the task.
        :param status: Current status of the task.
        """
        try:
            logging.info(f"[{self.name}] Task {task_id} progress: {status}")
        except Exception as e:
            logging.error(f"[{self.name}] Error tracking task progress: {e}")
    
    def retrieve_rag_context(self, query, top_k=5):
        """
        Retrieve context from RAG-based long-term memory.
        :param query: Query string for RAG.
        :param top_k: Number of results to retrieve.
        :return: List of relevant context or error message.
        """
        try:
            logging.info(f"[{self.name}] Querying RAG memory for: {query}")
            results = self.memory.retrieve_long_term_rag(query, top_k=top_k)
            if not results:
                logging.info(f"[{self.name}] No relevant context found in RAG memory.")
                return "No relevant context found."
            logging.info(f"[{self.name}] Retrieved RAG context: {results}")
            return results
        except Exception as e:
            logging.error(f"[{self.name}] Error retrieving RAG context: {e}")
            return f"Error retrieving context: {e}"


            
# MultiAgentSystem Class

class MultiAgentSystem:
    def __init__(self):
        """
        Initialize the Multi-Agent System.
        """
        self.agents = {}  # Dictionary of registered agents
        self.shared_memory = SharedMemory()  # Shared memory instance
        self.task_queue = []  # Task queue for dynamic delegation
        self.communication_hub = {}  # Message storage for inter-agent communication

    def register_agent(self, agent):
        """
        Register an agent in the system.
        :param agent: An instance of the Agent class.
        """
        if agent.name in self.agents:
            raise ValueError(f"Agent with name {agent.name} already exists!")
        self.agents[agent.name] = agent
        agent.shared_memory = self.shared_memory  # Provide shared memory access
        print(f"Agent {agent.name} registered successfully.")

    def add_task(self, task):
        """
        Add a task to the priority task queue.
        :param task: A dictionary or object representing the task.
        """
        priority = -task.get('priority', 0)  # Use negative for max-heap behavior
        heapq.heappush(self.task_queue, (priority, task))
        print(f"Task added to the queue: {task}")

    def assign_task(self):
        """
        Dynamically assign a task to an available agent.
        """
        if not self.task_queue:
            print("No tasks in the queue.")
            return None
        _, task = heapq.heappop(self.task_queue)
        available_agents = [agent for agent in self.agents.values() if not agent.is_busy]
        if not available_agents:
            print("No available agents to assign tasks.")
            return None

        # Assign task to the least busy agent
        least_busy_agent = min(available_agents, key=lambda x: len(x.memory.retrieve_short_term()))
        print(f"Assigning task '{task}' to agent {least_busy_agent.name}.")
        least_busy_agent.observe(task)
        return task
    
    def assign_task_with_rag(self, task, rag_query=None, top_k=5):
        """
        Assign a task to an available agent with optional RAG context retrieval.
        :param task: Task details.
        :param rag_query: Optional RAG query for augmenting context.
        :param top_k: Number of RAG results to retrieve.
        """
        if not self.task_queue:
            print("No tasks in the queue.")
            return None

        _, task = heapq.heappop(self.task_queue)
        available_agents = [agent for agent in self.agents.values() if not agent.is_busy]
        if not available_agents:
            print("No available agents to assign tasks.")
            return None

        # Assign task to the least busy agent
        least_busy_agent = min(available_agents, key=lambda x: len(x.memory.retrieve_short_term()))
        least_busy_agent.observe(task)

        # Retrieve RAG context if applicable
        if rag_query:
            rag_context = least_busy_agent.retrieve_rag_context(rag_query, top_k=top_k)
            if rag_context:
                task["rag_context"] = rag_context
                least_busy_agent.observe({"RAG_context": rag_context})
                print(f"Assigned RAG-augmented task to {least_busy_agent.name}: {task}")

        print(f"Assigning task '{task}' to agent {least_busy_agent.name}.")
        return task

    def send_message(self, sender, recipient, content):
        """
        Send a message from one agent to another.
        :param sender: Name of the sending agent.
        :param recipient: Name of the receiving agent.
        :param content: Message content.
        """
        if recipient not in self.agents:
            print(f"Recipient {recipient} does not exist.")
            return
        if recipient not in self.communication_hub:
            self.communication_hub[recipient] = []
        self.communication_hub[recipient].append({"from": sender, "content": content})
        print(f"Message sent from {sender} to {recipient}: {content}")

    def retrieve_messages(self, recipient):
        """
        Retrieve all messages for a specific agent.
        :param recipient: Name of the receiving agent.
        :return: List of messages.
        """
        messages = self.communication_hub.get(recipient, [])
        return sorted(messages, key=lambda x: x.get("priority", 0), reverse=True)

    def shared_context_summary(self):
        """
        Retrieve a summary of the shared memory for all agents.
        """
        return self.shared_memory.retrieve_all()
    
    def collaborate(self, agent_names, task):
        """
        Enable agents to collaborate on a task.
        :param agent_names: List of agent names involved in the collaboration.
        :param task: The task to be shared.
        """
        if not agent_names:
            print("No agents specified for collaboration.")
            return

        for agent_name in agent_names:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                agent.observe(task)
                print(f"Agent {agent_name} is collaborating on task: {task}")
            else:
                print(f"Agent {agent_name} is not registered in the system.")

    def collaborate_with_rag(self, agent_names, task, rag_query=None, top_k=5):
        """
        Enable agents to collaborate on a task with optional RAG context retrieval.
        :param agent_names: List of agent names involved in the collaboration.
        :param task: The task to be shared.
        :param rag_query: Query string for RAG context (if applicable).
        :param top_k: Number of RAG results to retrieve.
        """
        if not agent_names:
            print("No agents specified for collaboration.")
            return

        rag_context = None
        if rag_query:
            # Use the first agent to retrieve RAG context
            first_agent = self.agents.get(agent_names[0])
            if first_agent:
                rag_context = first_agent.retrieve_rag_context(rag_query, top_k=top_k)

        for agent_name in agent_names:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                collaboration_task = {"task": task}
                if rag_context:
                    collaboration_task["RAG_context"] = rag_context
                agent.observe(collaboration_task)
                print(f"Agent {agent_name} is collaborating on task: {collaboration_task}")
            else:
                print(f"Agent {agent_name} is not registered in the system.")

