# Data structures shared by ARM host and DSP

Some data structures need to be shared by the ARM host side and the DSP side.
Sharing might mean either the data is copied as one block from one side to the
other side or it might mean that the data structure lives in a memory block
that is accessed from both sides.

In either case, this means that ARM host and DSP need to use a common data
layout (type sizes and alignments). Because ARM and DSP data layouts are not
identical, the types defined in here use only fixed-width types (u?int##_t) and
all structs are packed to force identical alignment on both sides. This makes
access to the shared data structures slower than to natively aligned data
structures, so the use of such data structures must be limited.

When memory is copied, the address of the copy is different than the original
address. When memory is mapped to ARM host and DSP, it is usually mapped to
different addresses on each side. In both cases, this means that pointers do
not work. So all data structures defined in here must not use pointers.

Both ARM host and DSP use little endian, so at least, the data does not have to
be converted regarding endianness.

Using the data structures with the restrictions mentioned above boils down to
a kind of data serialization and deserialization. It is helpful to keep this in
mind when working with those data structures.

## Enumerations

C `enum` results in an `int` type (if all enum values can fit that type, but
that's usually the case). `int` does not have an architecture-independent width.
Thus, a fixed-width type has to be defined to store the values from the enum in
the data structures. By convention, the `enum` type is defined as
```
typedef enum myvals_enum { // "myvals" is a placeholder
    ...
} myvals_enum_t;
```
Additionally, a storage type for the values is defined:
```
typedef uint8/16/32_t myvals_store_t;
```
The storage type is to be used for struct members that are supposed to store
values of the enum:
```
typedef struct some_s {
    ...
    myvals_store_t myval;
    ...
} some_t;
```
In `switch` statements on the `enum`, the value is supposed to be casted to the
`..._enum_t` type, so the compiler warnings for left-out enum values trigger:
```
switch ((myvals_enum_t)some.myval) {
case ...:
    ...
}
```
