"""Tests for shared/data/ module."""

import pytest


class TestPairGenerator:
    """Tests for contrastive pair generation."""

    def test_generate_balanced_dataset(self):
        from shared.data.pair_generator import generate_arithmetic_dataset

        pairs = generate_arithmetic_dataset(n_pairs=100, seed=42)
        assert len(pairs) == 100
        assert all(p.pair_id for p in pairs)
        assert all(p.clean_prompt != p.corrupted_prompt for p in pairs)

    def test_all_task_types_present(self):
        from shared.data.pair_generator import generate_arithmetic_dataset, TASK_FACTORIES

        pairs = generate_arithmetic_dataset(n_pairs=100, seed=42)
        types_present = {p.pair_type for p in pairs}
        for tt in TASK_FACTORIES:
            factory_type = TASK_FACTORIES[tt](0, __import__("random").Random(0)).pair_type
            assert factory_type in types_present, f"Missing type: {factory_type}"

    def test_metadata_populated(self):
        from shared.data.pair_generator import generate_arithmetic_dataset

        pairs = generate_arithmetic_dataset(n_pairs=10, seed=42)
        for p in pairs:
            assert "a" in p.metadata or p.pair_type == "word_problem"

    def test_stratified_split(self):
        from shared.data.pair_generator import generate_arithmetic_dataset
        from shared.data.loaders import stratified_split

        pairs = generate_arithmetic_dataset(n_pairs=100, seed=42)
        train, val, test = stratified_split(pairs)
        assert len(train) + len(val) + len(test) == 100
        assert len(train) >= len(val)
        assert len(train) >= len(test)
