"""Unit tests for YAML config loader."""

import pytest

from aigp.utils.config_loader import load_config, merge_configs


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("key: value\nnested:\n  a: 1\n  b: 2\n")
        config = load_config(cfg_file)
        assert config["key"] == "value"
        assert config["nested"]["a"] == 1

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_empty_returns_empty_dict(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        config = load_config(cfg_file)
        assert config == {}


class TestMergeConfigs:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        merged = merge_configs(base, override)
        assert merged == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"nested": {"a": 1, "b": 2}, "top": "x"}
        override = {"nested": {"b": 3, "c": 4}}
        merged = merge_configs(base, override)
        assert merged["nested"] == {"a": 1, "b": 3, "c": 4}
        assert merged["top"] == "x"

    def test_empty_override(self):
        base = {"a": 1}
        merged = merge_configs(base, {})
        assert merged == {"a": 1}

    def test_does_not_mutate_base(self):
        base = {"a": 1}
        merge_configs(base, {"a": 2})
        assert base["a"] == 1
