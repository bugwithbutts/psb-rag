from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agents.config import Settings, get_settings


class SettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        get_settings.cache_clear()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(get_settings.cache_clear)
        self.addCleanup(self.env.stop)

    def test_default_limits(self) -> None:
        settings = get_settings()

        self.assertEqual(settings.max_retrieval_attempts, 5)
        self.assertEqual(settings.max_rewrite_subqueries_per_round, 4)
        self.assertEqual(settings.max_total_llm_calls, 30)
        self.assertEqual(settings.max_total_tokens_per_question, 40000)
        self.assertEqual(settings.max_total_tokens_soft, 30000)
        self.assertEqual(settings.history_limit, 8)

    def test_dataclass_defaults_match_get_settings(self) -> None:
        direct = Settings()
        resolved = get_settings()

        self.assertEqual(
            direct.max_retrieval_attempts,
            resolved.max_retrieval_attempts,
        )
        self.assertEqual(
            direct.max_rewrite_subqueries_per_round,
            resolved.max_rewrite_subqueries_per_round,
        )
        self.assertEqual(
            direct.max_total_llm_calls,
            resolved.max_total_llm_calls,
        )
        self.assertEqual(
            direct.max_total_tokens_per_question,
            resolved.max_total_tokens_per_question,
        )
        self.assertEqual(
            direct.max_total_tokens_soft,
            resolved.max_total_tokens_soft,
        )

    def test_env_overrides_limits(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MAX_REWRITE_SUBQUERIES_PER_ROUND": "1",
                "MAX_TOTAL_TOKENS_PER_QUESTION": "18000",
                "MAX_TOTAL_TOKENS_SOFT": "12000",
                "HISTORY_LIMIT": "6",
            },
        ):
            settings = get_settings()

        self.assertEqual(settings.max_rewrite_subqueries_per_round, 1)
        self.assertEqual(settings.max_total_tokens_per_question, 18000)
        self.assertEqual(settings.max_total_tokens_soft, 12000)
        self.assertEqual(settings.history_limit, 6)

    def test_soft_budget_must_be_lower_than_hard_budget(self) -> None:
        with self.assertRaises(ValueError):
            Settings(
                max_total_tokens_per_question=15000,
                max_total_tokens_soft=15000,
            )

    def test_positive_limits_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            Settings(max_rewrite_subqueries_per_round=0)
        with self.assertRaises(ValueError):
            Settings(max_retrieval_attempts=0)


if __name__ == "__main__":
    unittest.main()
