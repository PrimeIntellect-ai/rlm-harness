import unittest

from rlm.tools import SummarizeState, get_active_tools
from rlm.tools.base import ToolContext
from rlm.tools.ipython import IPYTHON_TIMEOUT_MAX_SECONDS, IpythonTool
from rlm.tools.summarize import SummarizeTool
from rlm.types import RLMMetrics, TokenUsage


class DummyRepl:
    def __init__(self):
        self.calls = []

    def execute(self, code, timeout=None):
        self.calls.append((code, timeout))
        return f"ran:{code}:{timeout}"


def make_context(messages=None, exec_timeout=17):
    return ToolContext(
        messages=messages or [],
        metrics=RLMMetrics(turns=4, turns_since_last_summarize=3),
        total_usage=TokenUsage(prompt_tokens=0, completion_tokens=12),
        last_prompt_tokens=20,
        exec_timeout=exec_timeout,
        repl=DummyRepl(),
        state={"summarize": SummarizeState()},
    )


class BuiltinToolTests(unittest.TestCase):
    def test_get_active_tools_includes_builtins(self):
        names = [tool["function"]["name"] for tool in get_active_tools()]
        self.assertEqual(names, ["ipython", "summarize"])

    def test_ipython_tool_uses_default_timeout_and_stringifies_code(self):
        tool = IpythonTool()
        context = make_context()

        outcome = tool.execute({"code": 123}, context)

        self.assertEqual(outcome.content, "ran:123:17")
        self.assertEqual(context.repl.calls, [("123", 17)])

    def test_ipython_tool_caps_requested_timeout_at_ten_minutes(self):
        tool = IpythonTool()
        context = make_context()

        outcome = tool.execute({"code": "print(1)", "timeout": 9999}, context)

        self.assertEqual(
            outcome.content,
            f"ran:print(1):{IPYTHON_TIMEOUT_MAX_SECONDS}",
        )
        self.assertEqual(
            context.repl.calls,
            [("print(1)", IPYTHON_TIMEOUT_MAX_SECONDS)],
        )

    def test_ipython_tool_caps_default_timeout_at_ten_minutes(self):
        tool = IpythonTool()
        context = make_context(exec_timeout=9999)

        outcome = tool.execute({"code": "print(1)"}, context)

        self.assertEqual(
            outcome.content,
            f"ran:print(1):{IPYTHON_TIMEOUT_MAX_SECONDS}",
        )
        self.assertEqual(
            context.repl.calls,
            [("print(1)", IPYTHON_TIMEOUT_MAX_SECONDS)],
        )

    def test_summarize_tool_updates_state_and_returns_string_content(self):
        tool = SummarizeTool()
        context = make_context(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a1"},
                {"role": "tool", "content": "t1"},
                {"role": "assistant", "content": "a2"},
            ]
        )

        outcome = tool.execute(
            {"num_turns": 1, "summary": {"x": 1}, "flush_repl_state": True},
            context,
        )

        self.assertEqual(outcome.content, "[turns 0-0] {'x': 1}")
        self.assertEqual(outcome.drop_turns, 1)
        self.assertIs(outcome.flush_repl_state, True)
        self.assertEqual(context.metrics.summarize_count, 1)
        self.assertEqual(context.metrics.summarize_total_turns_dropped, 1)
        self.assertEqual(context.metrics.summarize_summary_lengths, [8])
        self.assertEqual(context.state["summarize"].turn_at_last_summarize, 4)

    def test_summarize_tool_rejections_return_string_messages(self):
        tool = SummarizeTool()
        context = make_context(
            messages=[
                {"role": "assistant", "content": "a1"},
                {"role": "assistant", "content": "a2"},
            ]
        )

        outcome = tool.execute({"num_turns": 99, "summary": None}, context)

        self.assertIsInstance(outcome.content, str)
        self.assertIn("No context was dropped", outcome.content)
        self.assertEqual(context.metrics.summarize_rejected_count, 1)


if __name__ == "__main__":
    unittest.main()
