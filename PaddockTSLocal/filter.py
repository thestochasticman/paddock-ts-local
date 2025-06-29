from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Any, List, Union, Dict, Type
from marshmallow import fields

@dataclass_json
@dataclass(frozen=True)
class Filter:
    """
    Represents a STAC‐API filter expression node.

    Attributes:
        op (str): The operator, e.g. "and", "or", "<", "eq", "between", "not".
        args (List[Union[Filter, Dict, Any]]):
            The arguments to the operator. Each arg can be:
              - Another Filter (for nesting)
              - A literal dict like {"property": "eo:cloud_cover"}
              - A literal value (number or string)
    """
    op: str
    args: List[Union["Filter", Dict[str, Any], Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this Filter (and any nested Filters) into the raw dict
        form expected by pystac_client.

        Returns:
            Dict[str, Any]: The JSON-serializable filter expression.
        """
        serialized_args = []
        for a in self.args:
            if isinstance(a, Filter):
                serialized_args.append(a.to_dict())
            else:
                serialized_args.append(a)
        return {"op": self.op, "args": serialized_args}

    # Convenience constructors for common operations
    @classmethod
    def lt(cls: Type["Filter"], prop: str, value: Any) -> "Filter":
        """Less-than filter: property < value"""
        return cls("<", [{"property": prop}, value])

    @classmethod
    def eq(cls: Type["Filter"], prop: str, value: Any) -> "Filter":
        """Equality filter: property == value"""
        return cls("==", [{"property": prop}, value])

    @classmethod
    def between(cls: Type["Filter"], prop: str, interval: List[Any]) -> "Filter":
        """Between filter: property between [min, max]"""
        return cls("between", [{"property": prop}, interval])

    @classmethod
    def AND(cls: Type["Filter"], *filters: "Filter") -> "Filter":
        """Logical AND of multiple filters."""
        return cls("and", list(filters))

    @classmethod
    def OR(cls: Type["Filter"], *filters: "Filter") -> "Filter":
        """Logical OR of multiple filters."""
        return cls("or", list(filters))

    @classmethod
    def NOT(cls: Type["Filter"], filt: "Filter") -> "Filter":
        """Logical NOT of a single filter."""
        return cls("not", [filt])

    # Override bitwise operators for Pythonic chaining
    def __and__(self, other: "Filter") -> "Filter":
        return Filter.AND(self, other)

    def __or__(self, other: "Filter") -> "Filter":
        return Filter.OR(self, other)

    def __invert__(self) -> "Filter":
        return Filter.NOT(self)

    def __str__(self) -> str:
        return self.to_json(indent=2)
    

def test_filter_basic_and_logical():
    import json

    """Test Filter constructors, to_dict, and logical operators."""
    # Basic lt, eq, between
    f_lt = Filter.lt("eo:cloud_cover", 10)
    expected_lt = {"op": "lt", "args": [{"property": "eo:cloud_cover"}, 10]}
    assert f_lt.to_dict() == expected_lt

    f_eq = Filter.eq("eo:platform", "sentinel-2a")
    expected_eq = {"op": "eq", "args": [{"property": "eo:platform"}, "sentinel-2a"]}
    assert f_eq.to_dict() == expected_eq

    f_betw = Filter.between("datetime", ["2020-01-01", "2020-06-01"])
    expected_betw = {
        "op": "between",
        "args": [{"property": "datetime"}, ["2020-01-01", "2020-06-01"]]
    }
    assert f_betw.to_dict() == expected_betw

    # AND / OR / NOT via methods
    f1, f2 = f_lt, f_eq
    f_and = Filter.AND(f1, f2)
    assert f_and.op == "and" and f_and.args == [f1, f2]
    assert f_and.to_dict() == {"op": "and", "args": [expected_lt, expected_eq]}

    f_or = Filter.OR(f1, f2)
    assert f_or.op == "or" and f_or.args == [f1, f2]
    assert f_or.to_dict() == {"op": "or", "args": [expected_lt, expected_eq]}

    f_not = Filter.NOT(f1)
    assert f_not.op == "not" and f_not.args == [f1]
    assert f_not.to_dict() == {"op": "not", "args": [expected_lt]}

    # Operators &, |, ~
    assert (f1 & f2).to_dict() == f_and.to_dict()
    assert (f1 | f2).to_dict() == f_or.to_dict()
    assert (~f1).to_dict()      == f_not.to_dict()

    # Nested combination
    combo = (f1 & f2) | ~f1
    d = combo.to_dict()
    assert d["op"] == "or"
    assert isinstance(d["args"], list) and len(d["args"]) == 2
    assert d["args"][0]["op"] == "and"
    assert d["args"][1]["op"] == "not"

    # __str__ JSON matches to_dict
    str_json = json.loads(str(combo))
    assert str_json == d

if __name__ == '__main__':
    test_filter_basic_and_logical()