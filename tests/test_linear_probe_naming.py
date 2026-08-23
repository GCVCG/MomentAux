"""The probe's output filename is a contract: the default must stay
byte-identical to the recorded protocol (linear_probe.json), shots-only must
never write to that name (the 2026-08-06 clobber), and --out-suffix (block A
trajectory probes, 2026-08-23) goes LAST, immediately before .json."""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load():
    spec = importlib.util.spec_from_file_location(
        "lp", os.path.join(ROOT, "analysis", "linear_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_names_unchanged():
    lp = _load()
    assert lp.output_filename(False) == "linear_probe.json"
    assert lp.output_filename(False, None, "") == "linear_probe.json"
    assert lp.output_filename(True) == "linear_probe_shots.json"
    assert lp.output_filename(False, "cifar100super") == "linear_probe_cifar100super.json"
    assert lp.output_filename(True, "cifar100super") == "linear_probe_shots_cifar100super.json"


def test_out_suffix_goes_last():
    lp = _load()
    assert lp.output_filename(False, None, "_ep020") == "linear_probe_ep020.json"
    assert lp.output_filename(True, None, "_ep020") == "linear_probe_shots_ep020.json"
    assert lp.output_filename(False, "tin20", "_ep020") == "linear_probe_tin20_ep020.json"
    assert lp.output_filename(True, "tin20", "_ep020") == "linear_probe_shots_tin20_ep020.json"


def test_suffix_never_collides_with_default():
    lp = _load()
    for so in (False, True):
        assert lp.output_filename(so, None, "_x") != lp.output_filename(so)
