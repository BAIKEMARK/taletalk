import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_weclone_dataset import convert_sharegpt_rows, write_weclone_dataset


class PrepareWeCloneDatasetTests(unittest.TestCase):
    def test_convert_sharegpt_conversations_to_messages(self):
        rows = [
            {
                "system": "角色系统提示",
                "conversations": [
                    {"from": "human", "value": "问题"},
                    {"from": "gpt", "value": "回答"},
                ],
                "metadata": {"ignored": True},
            }
        ]

        converted = convert_sharegpt_rows(rows)

        self.assertEqual(
            converted,
            [
                {
                    "system": "角色系统提示",
                    "messages": [
                        {"role": "user", "content": "问题"},
                        {"role": "assistant", "content": "回答"},
                    ],
                }
            ],
        )

    def test_write_weclone_dataset_generates_dataset_info_and_stats(self):
        rows = [
            {
                "system": "角色系统提示",
                "conversations": [
                    {"from": "human", "value": "问题一"},
                    {"from": "gpt", "value": "回答一"},
                ],
            },
            {
                "system": "角色系统提示",
                "conversations": [
                    {"from": "human", "value": "问题二"},
                    {"from": "gpt", "value": "回答二"},
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            stats = write_weclone_dataset(rows, out_dir, dataset_name="chat-sft", limit=1)

            data = json.loads((out_dir / "sft-my.json").read_text(encoding="utf-8"))
            dataset_info = json.loads((out_dir / "dataset_info.json").read_text(encoding="utf-8"))
            written_stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))

        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["messages"][0]["role"], "user")
        self.assertEqual(dataset_info["chat-sft"]["formatting"], "sharegpt")
        self.assertEqual(dataset_info["chat-sft"]["columns"]["messages"], "messages")
        self.assertEqual(stats["written_examples"], 1)
        self.assertEqual(written_stats["source_examples"], 2)


if __name__ == "__main__":
    unittest.main()
