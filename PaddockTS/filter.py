from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Any, List, Union, Dict, Type
from marshmallow import fields
import re

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
    

    @classmethod
    def from_string(cls, s: str) -> "Filter":
        """
        Parse a simple string like:
        "eo:cloud_cover < 10 AND rain < 10"
        into a Filter object.

        Supported:
        - AND, OR, NOT (case-insensitive)
        - <, <=, >, >=, ==, !=
        - Parentheses for grouping
        """
        # Tokenize
        token_pattern = r"""
            (?P<LPAREN>\()
        | (?P<RPAREN>\))
        | (?P<AND>\bAND\b)
        | (?P<OR>\bOR\b)
        | (?P<NOT>\bNOT\b)
        | (?P<OP><=|>=|==|!=|<|>)
        | (?P<NUMBER>\d+(\.\d+)?)
        | (?P<STRING>'[^']*'|"[^"]*")
        | (?P<IDENT>[a-zA-Z0-9:_\-]+)
        | (?P<WS>\s+)
        """
        token_re = re.compile(token_pattern, re.VERBOSE | re.IGNORECASE)

        # Convert input string into list of tokens
        tokens = []
        for match in token_re.finditer(s):
            kind = match.lastgroup
            if kind == "WS":
                continue
            value = match.group()
            tokens.append((kind.upper(), value))
        tokens.append(("EOF", ""))

        # Recursive descent parser
        def parse_expression(index=0):
            def parse_primary(i):
                kind, value = tokens[i]
                if kind == "LPAREN":
                    node, i = parse_expression(i + 1)
                    if tokens[i][0] != "RPAREN":
                        raise SyntaxError("Expected ')'")
                    return node, i + 1
                elif kind == "NOT":
                    sub, i = parse_primary(i + 1)
                    return cls.NOT(sub), i
                elif kind == "IDENT":
                    # Expect IDENT OP VALUE
                    if tokens[i + 1][0] != "OP":
                        raise SyntaxError("Expected comparison operator after identifier")
                    op_token = tokens[i + 1][1]
                    value_token = tokens[i + 2]
                    prop = value
                    val_kind, val = value_token
                    if val_kind == "NUMBER":
                        val = float(val) if '.' in val else int(val)
                    elif val_kind == "STRING":
                        val = val.strip("\"'")
                    else:
                        raise SyntaxError("Invalid comparison value")
                    filter_obj = cls(op_token, [{"property": prop}, val])
                    return filter_obj, i + 3
                else:
                    raise SyntaxError(f"Unexpected token: {kind}")

            def parse_and_or(i):
                lhs, i = parse_primary(i)
                while i < len(tokens) and tokens[i][0] in ("AND", "OR"):
                    op = tokens[i][0]
                    rhs, i = parse_primary(i + 1)
                    if op == "AND":
                        lhs = cls.AND(lhs, rhs)
                    else:
                        lhs = cls.OR(lhs, rhs)
                return lhs, i

            return parse_and_or(index)

        result, final_index = parse_expression()
        if tokens[final_index][0] != "EOF":
            raise SyntaxError("Unexpected extra input after end of expression")
        return result

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
