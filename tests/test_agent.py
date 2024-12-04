import unittest
from reasonchain import Agent

class TestAgent(unittest.TestCase):
    def test_agent_initialization(self):
        agent = Agent(name="TestBot", model="test-model")
        self.assertEqual(agent.name, "TestBot")

    def test_agent_observe(self):
        agent = Agent(name="TestBot", model="test-model")
        input_data = "Sample input"
        self.assertEqual(agent.observe(input_data), "Sample input")
