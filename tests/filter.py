"""Test Filter class."""
import json
from PaddockTS.filter import Filter


def test_filter_lt():
    """Test less-than filter."""
    print("\n=== Testing Filter.lt ===")
    f = Filter.lt("eo:cloud_cover", 10)
    expected = {"op": "<", "args": [{"property": "eo:cloud_cover"}, 10]}
    assert f.to_dict() == expected
    print("[done] Filter.lt passed")


def test_filter_eq():
    """Test equality filter."""
    print("\n=== Testing Filter.eq ===")
    f = Filter.eq("eo:platform", "sentinel-2a")
    expected = {"op": "==", "args": [{"property": "eo:platform"}, "sentinel-2a"]}
    assert f.to_dict() == expected
    print("[done] Filter.eq passed")


def test_filter_between():
    """Test between filter."""
    print("\n=== Testing Filter.between ===")
    f = Filter.between("datetime", ["2020-01-01", "2020-06-01"])
    expected = {"op": "between", "args": [{"property": "datetime"}, ["2020-01-01", "2020-06-01"]]}
    assert f.to_dict() == expected
    print("[done] Filter.between passed")


def test_filter_and():
    """Test AND filter."""
    print("\n=== Testing Filter.AND ===")
    f1 = Filter.lt("eo:cloud_cover", 10)
    f2 = Filter.eq("eo:platform", "sentinel-2a")
    f_and = Filter.AND(f1, f2)

    assert f_and.op == "and"
    assert f_and.args == [f1, f2]
    assert f_and.to_dict() == {
        "op": "and",
        "args": [f1.to_dict(), f2.to_dict()]
    }
    print("[done] Filter.AND passed")


def test_filter_or():
    """Test OR filter."""
    print("\n=== Testing Filter.OR ===")
    f1 = Filter.lt("eo:cloud_cover", 10)
    f2 = Filter.eq("eo:platform", "sentinel-2a")
    f_or = Filter.OR(f1, f2)

    assert f_or.op == "or"
    assert f_or.args == [f1, f2]
    assert f_or.to_dict() == {
        "op": "or",
        "args": [f1.to_dict(), f2.to_dict()]
    }
    print("[done] Filter.OR passed")


def test_filter_not():
    """Test NOT filter."""
    print("\n=== Testing Filter.NOT ===")
    f1 = Filter.lt("eo:cloud_cover", 10)
    f_not = Filter.NOT(f1)

    assert f_not.op == "not"
    assert f_not.args == [f1]
    assert f_not.to_dict() == {"op": "not", "args": [f1.to_dict()]}
    print("[done] Filter.NOT passed")


def test_filter_bitwise_operators():
    """Test bitwise operator overloads (&, |, ~)."""
    print("\n=== Testing Filter bitwise operators ===")
    f1 = Filter.lt("eo:cloud_cover", 10)
    f2 = Filter.eq("eo:platform", "sentinel-2a")

    # & should be equivalent to AND
    assert (f1 & f2).to_dict() == Filter.AND(f1, f2).to_dict()

    # | should be equivalent to OR
    assert (f1 | f2).to_dict() == Filter.OR(f1, f2).to_dict()

    # ~ should be equivalent to NOT
    assert (~f1).to_dict() == Filter.NOT(f1).to_dict()

    print("[done] Filter bitwise operators passed")


def test_filter_nested():
    """Test nested filter combinations."""
    print("\n=== Testing Filter nested ===")
    f1 = Filter.lt("eo:cloud_cover", 10)
    f2 = Filter.eq("eo:platform", "sentinel-2a")

    combo = (f1 & f2) | ~f1
    d = combo.to_dict()

    assert d["op"] == "or"
    assert len(d["args"]) == 2
    assert d["args"][0]["op"] == "and"
    assert d["args"][1]["op"] == "not"
    print("[done] Filter nested passed")


def test_filter_to_json():
    """Test __str__ returns valid JSON matching to_dict."""
    print("\n=== Testing Filter __str__ ===")
    f1 = Filter.lt("eo:cloud_cover", 10)
    f2 = Filter.eq("eo:platform", "sentinel-2a")
    combo = f1 & f2

    str_json = json.loads(str(combo))
    assert str_json == combo.to_dict()
    print("[done] Filter __str__ passed")


def test_filter_from_string_simple():
    """Test parsing simple filter string."""
    print("\n=== Testing Filter.from_string simple ===")
    f = Filter.from_string("eo:cloud_cover < 10")
    assert f.op == "<"
    assert f.args == [{"property": "eo:cloud_cover"}, 10]
    print("[done] Filter.from_string simple passed")


def test_filter_from_string_and():
    """Test parsing AND filter string."""
    print("\n=== Testing Filter.from_string AND ===")
    f = Filter.from_string("eo:cloud_cover < 10 AND rain < 5")
    assert f.op == "and"
    assert len(f.args) == 2
    assert f.args[0].op == "<"
    assert f.args[1].op == "<"
    print("[done] Filter.from_string AND passed")


def test_filter_from_string_or():
    """Test parsing OR filter string."""
    print("\n=== Testing Filter.from_string OR ===")
    f = Filter.from_string("eo:cloud_cover < 10 OR rain < 5")
    assert f.op == "or"
    assert len(f.args) == 2
    print("[done] Filter.from_string OR passed")


def test_filter_from_string_not():
    """Test parsing NOT filter string."""
    print("\n=== Testing Filter.from_string NOT ===")
    f = Filter.from_string("NOT eo:cloud_cover < 10")
    assert f.op == "not"
    assert len(f.args) == 1
    assert f.args[0].op == "<"
    print("[done] Filter.from_string NOT passed")


def test_filter_from_string_parentheses():
    """Test parsing filter string with parentheses."""
    print("\n=== Testing Filter.from_string parentheses ===")
    f = Filter.from_string("(eo:cloud_cover < 10 AND rain < 5) OR snow > 0")
    assert f.op == "or"
    assert f.args[0].op == "and"
    print("[done] Filter.from_string parentheses passed")


def test_all():
    """Run all Filter tests."""
    print("=" * 50)
    print("Running Filter tests...")
    print("=" * 50)

    test_filter_lt()
    test_filter_eq()
    test_filter_between()
    test_filter_and()
    test_filter_or()
    test_filter_not()
    test_filter_bitwise_operators()
    test_filter_nested()
    test_filter_to_json()
    test_filter_from_string_simple()
    test_filter_from_string_and()
    test_filter_from_string_or()
    test_filter_from_string_not()
    test_filter_from_string_parentheses()

    print("\n" + "=" * 50)
    print("All Filter tests passed!")
    print("=" * 50)


if __name__ == '__main__':
    test_all()
