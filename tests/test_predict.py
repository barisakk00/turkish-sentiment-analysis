import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.predict import collect_texts, predict_sentiments, print_results
from src.text_processing import clean_text


class StubModel:
    def predict(self, texts):
        return ["Positive" for _ in texts]

    def predict_proba(self, texts):
        return [[0.05, 0.15, 0.80] for _ in texts]


class TextProcessingTests(unittest.TestCase):
    def test_clean_text_preserves_turkish_characters(self):
        result = clean_text("Çok İYİ! Güzel ürün: https://example.com/42")

        self.assertEqual(result, "çok iyi güzel ürün")

    def test_clean_text_removes_numbers_and_extra_spaces(self):
        result = clean_text("  Ürün  123   fena değil. ")

        self.assertEqual(result, "ürün fena değil")


class PredictionTests(unittest.TestCase):
    def test_collect_texts_combines_arguments_and_file_lines(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "inputs.txt"
            file_path.write_text("İkinci metin\n\nÜçüncü metin\n", encoding="utf-8")

            result = collect_texts([" Birinci metin "], file_path)

        self.assertEqual(result, ["Birinci metin", "İkinci metin", "Üçüncü metin"])

    def test_predict_sentiments_includes_confidence(self):
        result = predict_sentiments(StubModel(), ["Bu ürün çok güzel!"])

        self.assertEqual(result[0]["label"], "Positive")
        self.assertEqual(result[0]["confidence"], 0.80)
        self.assertEqual(result[0]["text"], "Bu ürün çok güzel!")

    def test_predict_sentiments_rejects_empty_cleaned_input(self):
        with self.assertRaisesRegex(ValueError, "became empty"):
            predict_sentiments(StubModel(), ["123 !!!"])

    def test_json_output_preserves_turkish_characters(self):
        output = io.StringIO()
        results = [{"text": "Güzel ürün", "label": "Positive", "confidence": 0.8}]

        with redirect_stdout(output):
            print_results(results, as_json=True)

        parsed = json.loads(output.getvalue())
        self.assertEqual(parsed, results)
        self.assertIn("Güzel ürün", output.getvalue())


if __name__ == "__main__":
    unittest.main()
