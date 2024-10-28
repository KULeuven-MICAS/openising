import pytest

from bqmpy.model import BinaryQuadraticModel, Vartype

class TestClass:
    
    @pytest.fixture
    def simple_bqm(self):
        def _simple_bqm(vartype=Vartype.SPIN):
            offset = 2
            linear = { 0:-1, 1:0, 2:-2, 3:0, 4:0, 5:2 }
            quadratic = {
                frozenset({0,1}):4, frozenset({0,2}):-3, frozenset({0,3}):8, frozenset({0,4}):1, frozenset({0,5}):1,
                frozenset({1,2}):1,
                frozenset({2,4}):-2, frozenset({2,5}):1,
                frozenset({4,5}):1
                         }
            return BinaryQuadraticModel(linear, quadratic, offset, vartype)
        return _simple_bqm

    @pytest.fixture
    def random_bqm(self):
        pass

    def test_construction(self):
        linear = { 0:1, 1:0, 2:-2 }
        quadratic = { frozenset({0,1}):4, frozenset({1,2}):-3 }
        offset = 2

        # spin model
        vartype = Vartype.SPIN
        bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype)
        for v, bias in linear.items():
            assert bqm.linear[v] == bias
        for v in bqm.linear:
            assert v in linear
        for e, bias in quadratic.items():
            assert bqm.quadratic[e] == quadratic[e]
        for e in bqm.quadratic:
            assert e in quadratic
        assert bqm.offset == offset
        assert bqm.vartype == vartype

        # binary model
        vartype = Vartype.BINARY
        bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype)
        for v, bias in linear.items():
            assert bqm.linear[v] == bias
        for v in bqm.linear:
            assert v in linear
        for e, bias in quadratic.items():
            assert bqm.quadratic[e] == quadratic[e]
        for e in bqm.quadratic:
            assert e in quadratic
        assert offset == bqm.offset
        assert bqm.vartype == vartype

        # vartype checking
        with pytest.raises(ValueError):
            vartype = "nonexistant type"
            bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype)
        with pytest.raises(ValueError):
            vartype = 1
            bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype)


    def test_shape(self, simple_bqm):
        bqm = simple_bqm()
        assert len(bqm) == 6
        assert bqm.num_variables == 6
        assert bqm.num_interactions == 9


    def test_eq(self, simple_bqm):
        bqm1 = simple_bqm()
        bqm2 = simple_bqm()
        assert bqm1 == bqm2


    def test_set_offset(self, simple_bqm):
        bqm = simple_bqm()
        assert bqm.offset == 2
        bqm.set_offset(0)
        assert bqm.offset == 0


    def test_set_variable(self, simple_bqm):
        # vartype=SPIN
        bqm = simple_bqm(Vartype.SPIN)
        assert 6 not in bqm.linear
        bqm.set_variable(6, 4)
        assert 6 in bqm.linear
        assert bqm.linear[6] == 4

        bqm.set_variable(6, 2)
        assert bqm.linear[6] == 2

        bqm.set_variable(6, -4, Vartype.BINARY)
        assert bqm.linear[6] == 2

        # vartype=BINARY
        bqm = simple_bqm(Vartype.BINARY)
        assert 6 not in bqm.linear
        bqm.set_variable(6, 4)
        assert 6 in bqm.linear
        assert bqm.linear[6] == 4

        bqm.set_variable(6, 2)
        assert bqm.linear[6] == 2

        bqm.set_variable(6, -4, Vartype.SPIN)
        assert bqm.linear[6] == 8


    def test_set_variables_from(self, simple_bqm):
        linear = { 5:5, 6:6, 7:7 }
        bqm = simple_bqm(Vartype.SPIN)
        bqm.set_variables_from(linear)
        for v, bias in linear.items():
            bqm.linear[v] == bias


    def test_set_interaction(self, simple_bqm):
        pass


    def test_set_interactions(self, simple_bqm):
        pass


    def test_remove_variable(self, simple_bqm):
        bqm = simple_bqm()
        v = 0
        assert v in bqm.linear
        assert any([v in e for e in bqm.quadratic])
        bqm.remove_variable(v)
        assert v not in bqm.linear
        assert not any([v in e for e in bqm.quadratic])


    def test_remove_interaction(self, simple_bqm):
        bqm = simple_bqm()
        e = frozenset({0,1})
        assert e in bqm.quadratic
        bqm.remove_interaction(e)
        assert e not in bqm.quadratic


    def test_scale(self, simple_bqm):
        bqm = simple_bqm()
        bqm2 = bqm.copy()
        scalar = 2
        bqm.scale(scalar)
        for v in bqm2.linear:
            assert bqm.linear[v] == scalar * bqm2.linear[v]
        for e in bqm2.quadratic:
            assert bqm.quadratic[e] == scalar * bqm2.quadratic[e]
        assert bqm.offset == scalar * bqm2.offset


    def test_copy(self, simple_bqm):
        bqm = simple_bqm()
        bqm2 = bqm.copy()
        assert bqm == bqm2
