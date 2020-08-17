## The LU decomposition
Given an *n* x *n* matrix *A*, when faced with the problem *A**x*** = ***b***, we have a system of *n* linear equations. This is rather tedious to solve, so a common approach is to break it down into two simpler problems by taking advantage of **Gaussian elimination**.

This is a process in which you can reduce a matrix *A* to an upper triangular matrix after a series of transformations. These transformations can be written in the form of a lower triangular matrix, hence factorising the matrix into two matrices *LU* = *A*, where *L* is the lower triangular matrix and *U* is the upper triangular matrix (both *n* x *n*).

The problem then becomes *LU**x*** = ***b***, which we can reduce to first solving  
*L**y*** = ***b***, and then  
*U**x*** = ***y***

As *L* and *U* are triangular matrices, both of these problems now just amount to one linear equation *n* times.

## Growth factor
The growth factor of a matrix is a measure of how accurately it can be factorised in floating point arithmetic.

The larger the growth factor, the more error prone the matrix is to floating point arithmetic.

# Investigation
First, we implement our own LU decomposition function, and also a function that solves the equation *A**x*** = ***b***.

Then, we'll compare our solve function with scipy's in-built function to see just how much more optimised it is.

Finally, we'll look at how the growth factor of a matrix behaves as we increase its dimensions.
