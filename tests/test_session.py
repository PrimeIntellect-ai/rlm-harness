import json
import tempfile
import unittest
from pathlib import Path

from rlm.session import Session


class SessionTests(unittest.TestCase):
    def test_log_tool_result_keeps_full_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session(Path(tmpdir) / "session")
            content = "x" * 5000

            session.log_tool_result(turn=3, tool="ipython", content=content, duration=1.234)
            session.close()

            messages_path = Path(tmpdir) / "session" / "messages.jsonl"
            entries = [json.loads(line) for line in messages_path.read_text().splitlines()]

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["type"], "tool_result")
            self.assertEqual(entries[0]["tool"], "ipython")
            self.assertEqual(entries[0]["content"], content)
            self.assertEqual(entries[0]["duration"], 1.234)


if __name__ == "__main__":
    unittest.main()
