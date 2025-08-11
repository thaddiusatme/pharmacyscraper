class APIError(Exception):
    """Base class for OpenAI API errors."""
    pass

class OpenAIError(APIError):
    """Base class for OpenAI errors (fallback)."""
    pass

class RateLimitError(APIError):
    pass

class Timeout(APIError):
    pass

class APIConnectionError(APIError):
    pass

class OpenAI:
    """Simple stub of the OpenAI client used in tests.
    The client provides a ``chat.completions.create`` method which can be
    patched by the test suite. The default implementation raises a
    ``NotImplementedError`` to make unexpected calls obvious.
    """
    def __init__(self, *args, **kwargs):
        self.chat = self._Chat()

    class _Chat:
        def __init__(self):
            self.completions = self._Completions()

        class _Completions:
            def create(self, *args, **kwargs):
                raise NotImplementedError("OpenAI chat.completions.create is not implemented in the stub")
