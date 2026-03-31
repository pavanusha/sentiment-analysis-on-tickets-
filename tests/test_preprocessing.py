import unittest

from ticket_sentiment.preprocessing import normalize_text


class PreprocessingTests(unittest.TestCase):
    def test_informal_english_is_normalized(self) -> None:
        text = "pls fix this asap, app is soooo buggy rn :("
        normalized = normalize_text(text)

        self.assertIn("please", normalized)
        self.assertIn("as", normalized)
        self.assertIn("soon", normalized)
        self.assertIn("possible", normalized)
        self.assertIn("application", normalized)
        self.assertIn("unstable", normalized)
        self.assertIn("right", normalized)
        self.assertIn("negative_emoji", normalized)

    def test_unicode_emoji_and_chat_slang_are_normalized(self) -> None:
        text = "thnx 🙏 works now 🙂"
        normalized = normalize_text(text)

        self.assertIn("thanks", normalized)
        self.assertIn("appreciate_emoji", normalized)
        self.assertIn("positive_emoji", normalized)
        self.assertIn("works", normalized)
        self.assertIn("now", normalized)


if __name__ == "__main__":
    unittest.main()
