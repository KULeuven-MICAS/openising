from abc import ABCMeta, abstractmethod
from typing import Any, Protocol, runtime_checkable
from collections.abc import Generator

class Stage(metaclass=ABCMeta):
    """! Abstract superclass for Runnables"""

    kwargs: dict[str, Any]

    def __init__(
        self,
        list_of_callables: list["StageCallable"],
        **kwargs: Any,
    ):
        """
        @param list_of_callables: a list of callables, that must have a signature compatible with this __init__ function
        and return a Stage instance. This is used to flexibly build iterators upon other iterators.
        @param kwargs: any keyword arguments, irrelevant to the specific class in question but passed on down
        """
        self.kwargs = kwargs
        self.list_of_callables = list_of_callables

    @abstractmethod
    def run(self) -> Generator: ...

    def __iter__(self):
        return self.run()

@runtime_checkable
class StageCallable(Protocol):
    def __call__(self, list_of_callables: list["StageCallable"], **kwagrs: Any) -> Stage: ...
