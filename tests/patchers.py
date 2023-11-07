# Copyright (c) 2020 Jeff Irion and contributors

"""Patches used for unit tests.

"""


from contextlib import contextmanager


class FileReadWrite(object):
    """Mock an opened file that can be read and written to."""

    def __init__(self):
        self._content = ""

    def read(self):
        """Mock reading from a file."""
        return self._content

    def readlines(self):
        """Mock reading line by line from a file."""
        for line in self._content.splitlines():
            yield line

    def write(self, content):
        """Mock writing to a file."""
        self._content += content

    def clear(self):
        """Clear the contents."""
        self._content = ""


FAKE_FILE = FileReadWrite()


@contextmanager
def open_fake_file(infile, mode="r"):  # pylint: disable=unused-argument
    """A context manager for mocking file I/O."""
    try:
        yield FAKE_FILE
    finally:
        pass
