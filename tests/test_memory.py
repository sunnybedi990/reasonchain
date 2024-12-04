import unittest
from reasonchain.memory import Memory

class TestMemory(unittest.TestCase):
    def test_short_term_memory(self):
        memory = Memory()
        memory.store_short_term("Test data")
        self.assertIn("Test data", memory.retrieve_short_term())

    def test_long_term_memory(self):
        memory = Memory()
        memory.store_long_term("key", "value")
        self.assertEqual(memory.retrieve_long_term("key"), "value")
        memory.clear_long_term()
        self.assertIsNone(memory.retrieve_long_term("key"))
