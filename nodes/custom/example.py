"""Example custom nodes — edit this file while the app is running.

The app polls this directory every second.  Save the file and new nodes
appear in the palette under the 'Custom' category without restarting.

Supported type hints:  float, int, bool, str, (none = ANY)
"""
from core.custom import node


@node(label="Lerp", category="Custom", description="Linear interpolation: a + t*(b-a)")
def lerp(a: float = 0.0, b: float = 1.0, t: float = 0.5) -> float:
    return a + t * (b - a)


@node(label="Clamp 01", category="Custom", description="Clamp value to [0, 1]")
def clamp_01(value: float = 0.5) -> float:
    return max(0.0, min(1.0, value))


@node(label="Sign", category="Custom", description="Returns -1, 0, or 1")
def sign(value: float = 0.0) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


@node(label="Word Count", category="Custom", description="Count words in a string")
def word_count(text: str = "hello world") -> int:
    return len(text.split())


@node(
    label="Min Max",
    category="Custom",
    description="Returns min and max of two values",
    outputs={"minimum": float, "maximum": float},
)
def min_max(a: float = 0.0, b: float = 1.0) -> dict:
    return {"minimum": min(a, b), "maximum": max(a, b)}
