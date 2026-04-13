"""Unit tests for ONNX policy inference engine."""

import pytest


class TestPolicyInferenceEngine:
    def test_missing_model_raises(self):
        from aigp.deployment.policy_inference import PolicyInferenceEngine
        with pytest.raises(FileNotFoundError):
            PolicyInferenceEngine(model_path="/nonexistent/policy.onnx")
