from dataclasses_json import dataclass_json, config
from datetime import date
from datetime import datetime
from marshmallow import fields
from os import makedirs as _makedirs

def parse_date(s: str) -> date:
    """
    Parse an ISO date string into a `date` object.

    Args:
        s (str): A date string in “YYYY-MM-DD” format.

    Returns:
        date: The corresponding `datetime.date` object.

    Raises:
        ValueError: If the string does not match the expected format.
    """
    if not isinstance(s, str): return s
    return datetime.strptime(s, "%Y-%m-%d").date()


def encode_date(d: date) -> str:
    """
    Encode a `date` object as an ISO date string.

    Args:
        d (date): The date to encode.

    Returns:
        str: The ISO-format date string (YYYY-MM-DD).
    """
    return d.isoformat()


def decode_date(s: str) -> date:
    """
    Decode an ISO date string into a `date` object.

    Args:
        s (str): A date string in “YYYY-MM-DD” format.

    Returns:
        date: The corresponding `datetime.date` object.
    """
    return date.fromisoformat(s)


# JSON (de)serialization config for date fields
date_config = config(
    encoder=encode_date,
    decoder=decode_date,
    mm_field=fields.Date
)

makedirs = lambda x: [_makedirs(x, exist_ok=True), x][1]