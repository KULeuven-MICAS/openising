import pathlib
import h5py


class HDF5Logger:
    def __init__(self, filename: pathlib.Path|None, schema: dict, buffer_size: int = 10000):
        """ Initialize the logger.

        The logger is responsible for writing to file in HDF5 format.
        It implements buffering to minimize (unnecessary) IO operations.
        If filename=None, the logger is run in feint mode and won't do anything.

        Parameters:
        - filename: Path to the HDF5 file. If filename=None, run the logger in feint mode.
        - schema: Dictionary mapping field names to h5py data types and optionally a shape.
                  Example: { "iteration": "i", "state": ("b", (10,)), "output": ("f4", (2, 2, 3)) }
        - buffer_size: Buffer size for batch writes.
        """
        if len(schema) == 0:
            raise ValueError("Schema should contain at least one entry")
        if buffer_size <= 0:
            raise ValueError("Buffer-size must be a strictly positive integer")

        self.filename = filename
        self.schema = schema
        self.buffer_size = buffer_size

        # Initialize attributes but do not open the file yet
        self.file = None  # `None` indicates the file is not yet open
        self.datasets = {}
        self.buffers = {}
        self.scalar_type = {}
        self.feint_mode = filename is None


    def __enter__(self):
        """ Enter the context manager. """
        self.open()
        return self  # Return the logger itself to be used inside the `with` block


    def __exit__(self, ex_type, ex_value, traceback):
        """ Close the context manager. """
        self.close()


    def open(self):
        """ Open the logger. """
        if self.feint_mode:
            return  # No-op

        if self.file is not None:
            raise RuntimeError("Logger is already active. Cannot enter context manager multiple times.")

        # Create the HDF5 file
        self.file = h5py.File(self.filename, "w")

        # Create datasets based on the schema.
        # The first dimension should be initialized with length 0 and unlimited size.
        # The other dimensions should match the given shape.
        self.datasets = {}
        for field, field_info in self.schema.items():
            dtype, shape = field_info if isinstance(field_info, tuple) else (field_info, None)

            if shape is None:  # Scalar value
                self.datasets[field] = self.file.create_dataset(
                    field, shape=(0,), maxshape=(None,), dtype=dtype
                )
                self.scalar_type[field] = True
            elif isinstance(shape, tuple):  # Non-scalar value
                self.datasets[field] = self.file.create_dataset(
                    field, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype
                )
                self.scalar_type[field] = False
            else:
                raise TypeError(f"Invalid shape for field '{field}'")

        # Initialize buffers for each field and set buffer_size
        self.buffers = { field: [] for field in self.schema.keys() }


    def write_metadata(self, **kwargs):
        """ Write metadata as attributes.

        Parameters:
        - kwargs: Key-value pairs to be written as attributes.
        """
        if self.feint_mode is True:
            return  # No-op

        if self.file is None:
            raise RuntimeError(
                "Cannot write metadata before opening the file. " + \
                "Please use the logger in a context manager (with statement) " + \
                "or manually open it first."
            )

        # Write metadata as attributes of the HDF5 file (root group).
        for key, value in kwargs.items():
            self.file.attrs[key] = value


    def log(self, **kwargs):
        """ Log values for the defined fields.

        Parameters:
        - kwargs: Key-value pairs matching the schema.
        """
        if self.feint_mode is True:
            return  # No-op

        if self.file is None:
            raise RuntimeError(
                "Cannot log data before opening the file. " + \
                "Please use the logger in a context manager (with statement) " + \
                "or manually open it first."
            )

        # write to buffers
        for field, value in kwargs.items():
            try:
                if self.scalar_type[field]:
                    self.buffers[field].append(value)
                else:
                    self.buffers[field].append(value.copy())
            except KeyError:
                raise KeyError(f"Field '{field}' was provided but is not defined in schema.")

        # Flush buffers if full (test on randomly chosen buffer
        if len(next(iter(self.buffers.values()))) >= self.buffer_size:
            self._flush_buffers()


    def _flush_buffers(self):
        """ Write buffered data to the HDF5 file."""
        # Write buffered data to file and empty out the buffers.
        for field in self.schema.keys():
            dset, buffer = self.datasets[field], self.buffers[field]
            if len(buffer) == 0:
                continue
            dset.resize(dset.shape[0] + len(buffer), axis=0)
            dset[-len(buffer):] = buffer
            buffer.clear()


    def close(self):
        """ Close the HDF5 file after all data is logged. """
        if self.feint_mode:
            return  # No-op

        if self.file is None:
            raise RuntimeError( "Cannot close a file which was not opened first.")

        # Write any remaining buffered data to file
        self._flush_buffers()

        # Close the file
        self.file.close()

        # Re-enable opening a file
        self.file = None
        self.datasets = {}
        self.buffers = {}
