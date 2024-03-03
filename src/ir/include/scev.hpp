/*
SCEV stands for Static Single Assignment (SSA) form Combined Expression Value.
It is a data structure used in compilers to represent expressions that vary
linearly with the execution of a loop.

Some key points about SCEV:

- It captures expressions that are affine functions of induction variables in
loops. This allows analyzing and optimizing loop-based code.

- The class contains a vector of Values that make up the expression. A Value can
be a constant, variable, operator, etc.

- It tracks BinaryInsts that are used to calculate elements of the expression.
This gives the instruction chain to generate the runtime values.

- It overloads operators like +, -, * to allow creating complex expressions
easily.

- Analysis passes can use SCEV to reason about and optimize loop trip counts,
increments, exit conditions etc.

- The "static single assignment" name comes from the fact that each SCEV
represents a pure function without side effects, allowing analysis as a
mathematical expression.

So in summary, SCEV is a IR-level representation of loop-based expressions to
help analyze and transform loops for better performance. The class allows
building these expressions conveniently.
*/
