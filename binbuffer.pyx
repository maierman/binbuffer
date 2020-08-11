
# the next line is not just a comment, it tells Cython to build C++ code
# distutils: language = c++

from cpython cimport Py_buffer
from cpython.buffer cimport PyBUF_SIMPLE, PyBUF_WRITEABLE
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdio cimport FILE, fread, fopen, fclose, fseek, SEEK_CUR
from libc.stdint cimport uint16_t
from cython.operator cimport dereference

"""
Here is our generic SimplestBuffer reimplemented with:

1) access restrictions ( no reallocating memory if the buffer has ever been accessed )
2) ability to preallocate memory if desired ( for some small speed gains )
3) method to read bytes directly from a file

NOTE:  If you examine SimpleBuffer below, you'll see that the view_count and buffer_accessed are somewhat redundant
and you could actually remove view_count entirerly without changing any functionality.  We've included
view_count for illustration purposes however, because it shows how reference counting is supposed to work
with the buffer protocol:  ie, if there are any open views on the buffer, then the buffer memory should not be changed
because other objects are referencing it.  Unfortunately, NumPy currently doesn't respect this protocol
and expects buffers to exist and be unchanged even after calls to releasebuffer.  For this reason, we added the bool
buffer_accessed to prevent any reallocation of buffer memory once a view on the memory has been requested
via getbuffer.

"""
cdef class SimpleBuffer:
    cdef: 
        vector[char] buf   # vector is useful here.  We get a contiguous block of memory but don't have to manage memory ourselves.
        int view_count        # reference counting for open views
        bool buffer_accessed  # we need this because NumPy expects buffers to exist even after releasebuffer

    def __cinit__(self):
        self.view_count = 0   
        self.buffer_accessed = False  
        
    def extend(self, b):
        self.add_bytes(b, len(b))
    
    def preallocate(self, total_size):
        if self.buffer_accessed or self.view_count > 0:
            raise RuntimeError('Buffer has been locked to changes in size')
        self.buf.reserve(total_size)
    
    cdef add_bytes(self, char *b, unsigned int n):
        if self.buffer_accessed or self.view_count > 0:
            raise RuntimeError('Buffer has been locked to changes in size')
        self.buf.insert(self.buf.end(), b, b + n)
        
    cdef add_bytes_from_file(self, FILE *fp, unsigned int n):
        if self.buffer_accessed or self.view_count > 0:
            raise RuntimeError('Buffer has been locked to changes in size')
        cdef int curr_size = self.buf.size()
        self.buf.resize(curr_size + n)
        fread(&self.buf[curr_size], 1, n, fp)
            
    def __getbuffer__(self, Py_buffer *buffer, int flags):
        if flags != PyBUF_SIMPLE and flags != PyBUF_SIMPLE | PyBUF_WRITEABLE:
            raise BufferError
            
        buffer.buf = &self.buf[0]
        buffer.format = NULL                    # NULL format means bytes 
        buffer.internal = NULL                  # see References
        buffer.itemsize = 1
        buffer.len = self.buf.size()
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = not (flags & PyBUF_WRITEABLE)
        buffer.shape = NULL
        buffer.strides = NULL
        buffer.suboffsets = NULL    
        
        self.view_count += 1
        self.buffer_accessed = True

    def __releasebuffer__(self, Py_buffer *buffer):
        self.view_count -= 1  
        

def fan_bytes(bytes input_bytes, SimpleBuffer buf1, SimpleBuffer buf2):
    cdef int num_bytes = len(input_bytes)
    cdef char *b = <char *>input_bytes  # you can cast bytes objects to char *
    cdef int cursor = 0
    cdef uint16_t msg_type
    cdef uint16_t msg_len
    
    # here we step through the character array by doing some C/C++ pointer arithmetic
    while cursor < num_bytes:
        body_len = dereference(<uint16_t*>(b + cursor)) 
        msg_type = dereference(<uint16_t*>(b + cursor + 2))
        
        if msg_type == 1:
            buf1.add_bytes(b + cursor, body_len + 4)  # msg_len + 4 is our total record length including the 4 byte header
        elif msg_type == 2:
            buf2.add_bytes(b + cursor, body_len + 4)
            
        cursor += body_len + 4


def fan_binary_file(bytes filename, SimpleBuffer buf1, SimpleBuffer buf2):
    cdef uint16_t header[2]
    cdef FILE *fp = fopen(filename, "r")
    while fread(header, 1, 4, fp) == 4:
        if header[1] == 1:
            buf1.add_bytes(<char *>header, 4)
            buf1.add_bytes_from_file(fp, header[0])
        elif header[1] == 2:
            buf2.add_bytes(<char *>header, 4)
            buf2.add_bytes_from_file(fp, header[0])
        else:
            fseek(fp, header[0], SEEK_CUR)
    
    fclose(fp)
