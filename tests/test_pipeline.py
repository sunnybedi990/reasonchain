import unittest
from reasonchain import Agent, CoTPipeline

class TestCoTPipeline(unittest.TestCase):
    def test_pipeline_steps(self):
        agent = Agent(name="TestBot", model="test-model")
        pipeline = CoTPipeline(agent=agent)
        pipeline.add_step("Step 1")
        self.assertEqual(len(pipeline.steps), 1)

    def test_pipeline_execution(self):
        agent = Agent(name="TestBot", model="test-model")
        pipeline = CoTPipeline(agent=agent)
        pipeline.add_step("Step 1")
        pipeline.add_step("Step 2")
        result = pipeline.execute()
        self.assertIn("Processed: Step 1", result)
