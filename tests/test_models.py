import inspect

import pytest
import torch

from models import FullBandTeacher, RGBHighFrequencyStudent


def _inputs(batch=2):
    return (
        torch.rand(batch, 3, 64, 64), torch.rand(batch, 3, 16, 16),
        torch.rand(batch, 9, 32, 32), torch.rand(batch, 9, 16, 16),
    )


def test_teacher_and_student_are_binary_evidence_models():
    rgb, ll2, h1, h2 = _inputs()
    teacher = FullBandTeacher(pretrained=False, embedding_dim=32).eval()
    student = RGBHighFrequencyStudent(pretrained=False, embedding_dim=32).eval()
    with torch.inference_mode():
        teacher_output = teacher(rgb, ll2, h1, h2)
        student_output = student(rgb, h1, h2)
    assert teacher_output.logits.shape == (2, 2)
    assert teacher_output.low_logits.shape == (2, 2)
    assert student_output.logits.shape == (2, 2)
    assert student_output.rgb_logits.shape == (2, 2)
    assert student_output.high_logits.shape == (2, 2)
    assert student_output.subclass_logits.shape == (2, 2)
    assert student_output.embedding.shape == (2, 32)
    assert student_output.subclass_embedding.shape == (2, 32)
    assert student_output.embedding.data_ptr() != student_output.subclass_embedding.data_ptr()


def test_student_has_no_ll2_argument():
    parameters = list(inspect.signature(RGBHighFrequencyStudent.forward).parameters)
    assert "ll2" not in parameters
    rgb, ll2, h1, h2 = _inputs()
    with pytest.raises(TypeError):
        RGBHighFrequencyStudent(pretrained=False)(rgb, ll2, h1, h2, False)
