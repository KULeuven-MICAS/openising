import math

class clock:
    """Imitates the internal clock of a CPU.

    Args:
        frequency (float): the frequency of the clock cycle.
        nb_operations (int): the amount of operations the clock can perform in one cycle.
        cycle_count (int): amount of cycles that have passed since creation.
    """
    def __init__(self, frequency:float, nb_operations:int):
        self.frequency = frequency
        self.nb_operations = nb_operations
        self.operation_count = 0
        self.cycle_count = 0

    def add_operations(self, operations:int):
        """Adds the operations to the clock.

        Args:
            operations (int): amount of operations to add
        """
        self.operations += operations

    def perform_operations(self) -> float:
        """Performs the given amount of operations and returns the time passed since creation.

        Args:
            operations (int): the operations that have been performed
        Returns:
            time (float): the time after performing the operations
        """
        self.cycle_count += math.ceil(self.operations / self.nb_operations)
        self.operations = 0
        return self.get_time()


    def get_time(self)-> float:
        """Get time of the clock at a given moment without adding operations.

        Returns:
            time (float): time at the current amount of cycles.
        """
        return self.cycle_count / self.frequency

    def add_cycles(self, cycles:int):
        """Adds the given amount of clock cycles to the clock and returns the time.

        Args:
            cycles (int): the amount of cycles to add
        Returns:

        """
        self.cycle_count += cycles
        return self.get_time()
