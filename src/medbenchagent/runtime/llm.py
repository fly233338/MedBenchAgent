"""Context budget error shared by native Agent request guards."""

class RequestContextBudgetError(ValueError):
    """The complete serialized request cannot fit without losing required context."""

    def __init__(self, actual: int, limit: int):
        super().__init__(f"Model request requires {actual} characters, exceeding context_chars={limit}; "
                         "narrow the evidence query or increase the context budget. Required context was not truncated.")
        self.diagnostics = {"request_chars": actual, "context_chars": limit}
