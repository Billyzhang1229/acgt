"""The contract in core.py agrees with the spec and with the reference tools.

Sentinel values are compared bit for bit against vcztools, the reference
reader; if these drift, our stores stop being readable by anything else.
"""

import numpy as np
import pytest
from vcztools import constants as vcz_constants

from acgt import core


def bits(x):
    """The raw bit pattern of a float scalar, at its own width."""
    a = np.array([x])
    return int(a.view(f"i{a.dtype.itemsize}")[0])


class TestSentinels:
    def test_int(self):
        assert core.INT_MISSING == vcz_constants.INT_MISSING
        assert core.INT_FILL == vcz_constants.INT_FILL

    def test_str(self):
        assert core.STR_MISSING == vcz_constants.STR_MISSING
        assert core.STR_FILL == vcz_constants.STR_FILL

    def test_float32(self):
        assert bits(core.FLOAT32_MISSING) == vcz_constants.FLOAT32_MISSING_AS_INT32
        assert bits(core.FLOAT32_FILL) == vcz_constants.FLOAT32_FILL_AS_INT32

    def test_float16(self):
        assert bits(core.FLOAT16_MISSING) == bits(vcz_constants.FLOAT16_MISSING)
        assert bits(core.FLOAT16_FILL) == bits(vcz_constants.FLOAT16_FILL)

    def test_float64(self):
        assert bits(core.FLOAT64_MISSING) == bits(vcz_constants.FLOAT64_MISSING)
        assert bits(core.FLOAT64_FILL) == bits(vcz_constants.FLOAT64_FILL)

    @pytest.mark.parametrize(
        ("missing", "fill"),
        [
            (core.FLOAT16_MISSING, core.FLOAT16_FILL),
            (core.FLOAT32_MISSING, core.FLOAT32_FILL),
            (core.FLOAT64_MISSING, core.FLOAT64_FILL),
        ],
    )
    def test_floats_are_nan(self, missing, fill):
        # The sentinels must be NaNs so numeric code treats them as absent,
        # and must differ from the ordinary NaN that arithmetic produces.
        assert np.isnan(missing)
        assert np.isnan(fill)
        assert bits(missing) != bits(type(missing)("nan"))

    def test_floats_do_not_survive_casting(self):
        # Widening the float32 sentinel does not produce the float64 one,
        # which is why every width needs its own constant.
        assert bits(np.float64(core.FLOAT32_MISSING)) != bits(core.FLOAT64_MISSING)


class TestFixedArrays:
    def test_required_arrays_are_specified(self):
        assert set(core.FIXED_ARRAYS) >= core.REQUIRED_ARRAYS

    def test_reserved_variable_names_match_vcztools(self):
        assert set(vcz_constants.RESERVED_VARIABLE_NAMES) <= set(core.FIXED_ARRAYS)

    def test_dims_are_reserved(self):
        for name, (dims, _kinds) in core.FIXED_ARRAYS.items():
            for dim in dims:
                assert dim in core.RESERVED_DIMS, (name, dim)
