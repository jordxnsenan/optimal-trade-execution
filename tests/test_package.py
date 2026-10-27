import unittest
import execai

class PackageImportTests(unittest.TestCase):

    def test_execai_can_be_imported(self):
        self.assertIsNotNone(execai.__file__)

if __name__ == "__main__":
    unittest.main()