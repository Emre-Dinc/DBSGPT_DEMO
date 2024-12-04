import unittest
from src.llm.query_handler import QueryHandler

class TestLiveQueries(unittest.TestCase):
    def setUp(self):
        self.handler = QueryHandler()

    def test_library_query(self):
        result = self.handler.process_query("student card?")
        print(f"\nQuery: student card?")
        print(f"Response: {result['response']}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
