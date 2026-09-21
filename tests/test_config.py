"""Tests for the configuration system."""

import json

import pytest
import yaml

from fedxcrop.config import (
    ExperimentConfig,
    config_hash,
    library_versions,
    load_config,
    save_run_metadata,
    to_dict,
)

BASE = "configs/base.yaml"


def test_base_config_loads_into_dataclasses():
    cfg = load_config(BASE)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.model.num_classes == 38
    assert cfg.data.image_size == 224
    assert cfg.federated.init == "imagenet"
    assert cfg.xai.methods == ["gradcam", "smoothgrad"]


def test_split_fractions_sum_to_one():
    cfg = load_config(BASE)
    total = cfg.data.train_fraction + cfg.data.val_fraction + cfg.data.test_fraction
    assert total == pytest.approx(1.0)


def test_dotted_overrides_apply_and_coerce_types():
    cfg = load_config(BASE, ["federated.rounds=5", "partition.alpha=0.1", "name=demo"])
    assert cfg.federated.rounds == 5
    assert isinstance(cfg.federated.rounds, int)
    assert cfg.partition.alpha == pytest.approx(0.1)
    assert cfg.name == "demo"


def test_override_of_optional_field_accepts_null():
    cfg = load_config(BASE, ["federated.train_subset_size=5000"])
    assert cfg.federated.train_subset_size == 5000
    cfg = load_config(BASE, ["federated.train_subset_size=null"])
    assert cfg.federated.train_subset_size is None


def test_unknown_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nnot_a_real_key: 1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(bad)


def test_malformed_override_is_rejected():
    with pytest.raises(ValueError, match="key.sub=value"):
        load_config(BASE, ["federated.rounds"])


def test_base_inheritance_merges_nested_keys(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text("base: base.yaml\nname: child\nfederated:\n  strategy: fedprox\n")
    (tmp_path / "base.yaml").write_text(
        "name: parent\nfederated:\n  strategy: fedavg\n  rounds: 30\n  mu: 0.01\n"
    )
    cfg = load_config(child)
    assert cfg.name == "child"
    assert cfg.federated.strategy == "fedprox"
    # Keys the child did not mention are inherited, not reset to the dataclass default.
    assert cfg.federated.rounds == 30
    assert cfg.federated.mu == pytest.approx(0.01)


def test_config_hash_is_stable_and_sensitive():
    a = load_config(BASE)
    b = load_config(BASE)
    assert config_hash(a) == config_hash(b)

    c = load_config(BASE, ["federated.rounds=31"])
    assert config_hash(a) != config_hash(c)

    # Output locations and worker count do not change the science, so they do
    # not change the hash.
    d = load_config(BASE, ["runs_dir=elsewhere", "data.num_workers=1"])
    assert config_hash(a) == config_hash(d)


def test_save_run_metadata_records_provenance(tmp_path):
    cfg = load_config(BASE, ["name=provenance_check"])
    meta_path = save_run_metadata(tmp_path, cfg)

    metadata = json.loads(meta_path.read_text())
    assert metadata["config_hash"] == config_hash(cfg)
    assert set(metadata["git"]) == {"commit", "dirty"}
    assert "torch" in metadata["libraries"]
    assert metadata["device"]

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert saved == to_dict(cfg)


def test_library_versions_reports_core_packages():
    versions = library_versions()
    for name in ("python", "torch", "numpy"):
        assert name in versions
