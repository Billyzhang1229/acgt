"""The schema in core.py agrees with the spec and with the reference tools.

Sentinel values are compared bit for bit against vcztools, the reference
reader; if these drift, our stores stop being readable by anything else.
"""

import json

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
        for spec in core.FIXED_ARRAYS.values():
            for dim in spec.dims:
                assert dim in core.RESERVED_DIMS, (spec.name, dim)

    def test_kinds_are_valid(self):
        for spec in core.FIXED_ARRAYS.values():
            assert spec.kind in core.KINDS, spec.name

    def test_names_match_keys(self):
        for name, spec in core.FIXED_ARRAYS.items():
            assert spec.name == name

    def test_region_index_has_six_columns(self):
        assert len(core.REGION_INDEX_COLUMNS) == 6


class TestSpecForField:
    def test_info_scalar(self):
        s = core.spec_for_field("INFO", "DP", "1", "Integer")
        assert s == core.ArraySpec("variant_DP", core.KIND_INT, ("variants",))

    def test_info_per_alt(self):
        s = core.spec_for_field("INFO", "AF", "A", "Float")
        assert s.name == "variant_AF"
        assert s.dims == ("variants", "alt_alleles")
        assert s.kind == core.KIND_FLOAT

    def test_format_per_allele(self):
        s = core.spec_for_field("FORMAT", "AD", "R", "Integer")
        assert s.name == "call_AD"
        assert s.dims == ("variants", "samples", "alleles")

    def test_format_per_genotype(self):
        s = core.spec_for_field("FORMAT", "PL", "G", "Integer")
        assert s.dims == ("variants", "samples", "genotypes")

    def test_flag(self):
        s = core.spec_for_field("INFO", "DB", "0", "Flag")
        assert s.kind == core.KIND_BOOL
        assert s.dims == ("variants",)

    def test_unbounded_number_gets_field_dim(self):
        s = core.spec_for_field("INFO", "ANN", ".", "String")
        assert s.dims == ("variants", "INFO_ANN_dim")

    def test_fixed_count_gets_field_dim(self):
        s = core.spec_for_field("FORMAT", "XX", "2", "Integer")
        assert s.dims == ("variants", "samples", "FORMAT_XX_dim")

    def test_character_is_its_own_kind(self):
        s = core.spec_for_field("INFO", "C", "1", "Character")
        assert s.kind == core.KIND_CHAR
        assert s.kind != core.spec_for_field("INFO", "S", "1", "String").kind

    def test_info_name_colliding_with_fixed_array_is_rejected(self):
        with pytest.raises(ValueError, match="variant_position"):
            core.spec_for_field("INFO", "position", "1", "String")

    def test_format_name_colliding_with_fixed_array_is_rejected(self):
        with pytest.raises(ValueError, match="call_genotype"):
            core.spec_for_field("FORMAT", "genotype", "1", "String")

    def test_gt_is_rejected(self):
        with pytest.raises(ValueError, match="call_genotype"):
            core.spec_for_field("FORMAT", "GT", "1", "String")

    def test_bad_category_is_rejected(self):
        with pytest.raises(ValueError, match="category"):
            core.spec_for_field("FILTER", "PASS", "1", "String")

    def test_bad_type_is_rejected(self):
        with pytest.raises(ValueError, match="Type"):
            core.spec_for_field("INFO", "X", "1", "Whatever")


class TestSchema:
    def make(self):
        fields = (
            *core.FIXED_ARRAYS.values(),
            core.spec_for_field("INFO", "DP", "1", "Integer"),
            core.spec_for_field("INFO", "AF", "A", "Float"),
        )
        dims = {
            "variants": 100,
            "samples": 1,
            "ploidy": 2,
            "alleles": 4,
            "alt_alleles": 3,
            "contigs": 25,
            "filters": 1,
            "region_index_values": 12,
            "region_index_fields": 6,
        }
        return core.Schema(dims=dims, fields=fields, source="vcf", build="GRCh38")

    def test_fixture_declares_every_dimension(self):
        # make() covers all of FIXED_ARRAYS, so constructing it also proves
        # the fixed arrays only use dimensions a real store would declare.
        self.make()

    def test_undeclared_dimension_is_rejected(self):
        field = core.spec_for_field("INFO", "ANN", ".", "String")
        with pytest.raises(ValueError, match="INFO_ANN_dim"):
            core.Schema(dims={"variants": 1}, fields=(field,))

    def test_roundtrips_through_json(self):
        schema = self.make()
        back = core.Schema.fromdict(json.loads(json.dumps(schema.asdict())))
        assert back == schema

    def test_version_mismatch_is_rejected(self):
        d = self.make().asdict()
        d["schema_version"] = "0.0"
        with pytest.raises(ValueError, match="version"):
            core.Schema.fromdict(d)

    def test_field_map(self):
        schema = self.make()
        m = schema.field_map()
        assert m["variant_DP"].kind == core.KIND_INT
        assert m["variant_position"].dims == ("variants",)
