import pytest
from hallucination_detector import detect

CASES = [
    ("Revenue was 15 million.", "Revenue was 15 million.", "SUPPORTED"),
    ("Revenue was 15 million.", "Revenue was 20 million.", "CONTRADICTED"),
    ("Revenue was 15 million.", "The company has 200 employees.", "NOT_IN_CONTEXT"),
    ("The company increased revenue.", "The company decreased revenue.", "CONTRADICTED"),
    ("The company increased revenue.", "The company increased revenue.", "SUPPORTED"),
    ("The company increased revenue.", "The company expanded its revenue.", "NOT_IN_CONTEXT"),
    ("The product is safe.", "The product is not safe.", "CONTRADICTED"),
    ("The product is not safe.", "The product is safe.", "CONTRADICTED"),
    ("All products passed inspection.", "Some products passed inspection.", "SUPPORTED"),
    ("Some products passed inspection.", "All products passed inspection.", "SUPPORTED"),
    ("Sales rose by 10%.", "Sales rose by 10%.", "SUPPORTED"),
    ("Sales rose by 10%.", "Sales rose by 15%.", "CONTRADICTED"),
    ("Sales rose by 10%.", "Profit rose by 10%.", "NOT_IN_CONTEXT"),
    ("The office opened in 2020.", "The office opened in 2020.", "SUPPORTED"),
    ("The office opened in 2020.", "The office opened in 2022.", "CONTRADICTED"),
    ("The office opened in 2020.", "The office has 50 rooms.", "NOT_IN_CONTEXT"),
    ("The company employs 50 people.", "The company employs 50 people.", "SUPPORTED"),
    ("The company employs 50 people.", "The company employs 60 people.", "CONTRADICTED"),
    ("The company employs 50 people.", "The company exports to Spain.", "NOT_IN_CONTEXT"),
    ("The door is open.", "The door is closed.", "CONTRADICTED"),
    ("The door is open.", "The door is open.", "SUPPORTED"),
    ("The door is open.", "The building is old.", "NOT_IN_CONTEXT"),
    ("The company gained profit.", "The company lost profit.", "CONTRADICTED"),
    ("The company gained profit.", "The company gained profit.", "SUPPORTED"),
    ("The company gained profit.", "The company has 12 offices.", "NOT_IN_CONTEXT"),
    ("The shipment arrived on Monday.", "The shipment arrived on Monday.", "SUPPORTED"),
    ("The shipment arrived on Monday.", "The shipment arrived on Friday.", "NOT_IN_CONTEXT"),
    ("The shipment arrived on Monday.", "The shipment never arrived.", "CONTRADICTED"),
    ("The store is open.", "The store is not open.", "CONTRADICTED"),
    ("The store is open.", "The store sells 500 products.", "NOT_IN_CONTEXT"),
]


@pytest.mark.parametrize("context,answer,expected", CASES)
def test_benchmark_case(context, answer, expected):
    result = detect(context, answer)
    actual = result.claims[0].label.value
    assert actual == expected, (
        f"Expected {expected}, got {actual}; "
        f"context={context!r}, answer={answer!r}"
    )
