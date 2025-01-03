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
        self.cycle_count = 0
    

    def perform_operations(self, operations: int) -> float:
        """Performs the given amount of operations and returns the time passed since creation.

        Args:
            operations (int): the operations that have been performed
        Returns:
            time (float): the time after performing the operations
        """
        self.cycle_count += math.ceil(operations / self.nb_operations)
        return self.cycle_count / self.frequency
    
    
    def get_time(self)-> float:
        """Get time of the clock at a given moment without adding operations.

        Returns:
            time (float): time at the current amount of cycles.
        """
        return self.cycle_count / self.frequency
