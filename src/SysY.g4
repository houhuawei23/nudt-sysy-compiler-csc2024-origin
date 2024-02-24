/*===-------------------------------------------===*/
/* SysY.g4 CREATED BY TXS 2024-2-16                */
/*===-------------------------------------------===*/
grammar SysY;
/*===-------------------------------------------===*/
/* Auxillary rules                                 */
/*===-------------------------------------------===*/
COMMA: ',';
CONSTLABEL: 'const';
INTTYPE: 'int';
FLOATTYPE: 'float';
SEMICOLON: ';';
L_PARENTHESIS: '(';
R_PARENTHESIS: ')';
L_BRACKET: '[';
R_BRACKET: ']';
L_BRACE: '{';
R_BRACE: '}';
ASSIGNMARK: '=';
VOIDTYPE: 'void';
IFKEY: 'if';
ELSEKEY: 'else';
WHILEKEY: 'while';
BREAKKEY: 'break';
CONTINUEKEY: 'continue';
RETURNKEY: 'return';
ADDOP: '+';
MINUSOP: '-';
NOTOP: '!';
MULOP: '*' | '/' | '%';
RELOP: '>' | '<' | '>=' | '<=';
EQOP: '==' | '!=';
LANDOP: '&&';
LOROP: '||';

/*===-------------------------------------------===*/
/* Lexer rules                                     */
/*===-------------------------------------------===*/

/*sysy-2022-spec-P3 1.Ident*/
fragment Identifier_nondigit: [_a-zA-Z];
fragment Digit: [0-9];
/* 
original version is left-recursive 
IDENTIFIER:   Identifier_nondigit 
			| IDENTIFIER Identifier_nondigit 
			| IDENTIFIER Digit;
*/

IDENTIFIER: Identifier_nondigit (Identifier_nondigit | Digit)*;

/*sysy-2022-spec-P4 3.valueConst*/
/*IntConst*/
fragment Hexadecimal_prefix: '0x' | '0X';
fragment Nonzero_digit: [1-9];
fragment Octal_digit: [0-7];
fragment Hexadecimal_digit: [0-9a-fA-F];

/*
original version is left recursive	
DECIMAL_CONST: Nonzero_digit | DECIMAL_CONST Digit;
*/
fragment Decimal_const: Nonzero_digit Digit*;

 
/* original version is left recursive       */
/* OCTAL_CONST: '0' | OCTAL_CONST Octal_digit; */

fragment Octal_const: '0' Octal_digit*;

//original version is left recursive 
/* HEXADECIMAL_CONST: Hexadecimal_prefix Hexadecimal_digit
					| HEXADECIMAL_CONST Hexadecimal_digit;*/
fragment Hexadecimal_const:
	Hexadecimal_prefix Hexadecimal_digit+;

INTEGER_CONST: Decimal_const | Octal_const | Hexadecimal_const;

/*FloatConst*/
/*http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1124.pdf
 P57-Floating Constants
 */
fragment Digit_sequence: Digit+;
fragment Sign: [+-];
fragment Hexadecimal_digit_sequence: Hexadecimal_digit+;
fragment Binary_exponent_part: ('P' | 'p') Sign? Digit_sequence;
fragment Exponent_part: ('E' | 'e') Sign? Digit_sequence;

/*
hexadecimal-fractional-constant:
	hexadecimal-digit-sequence_opt . hexadecimal-digit-sequence
	| hexadecimal-digit-sequence .
 */
fragment Hexadecimal_fractional_const:
	Hexadecimal_digit_sequence '.' Hexadecimal_digit_sequence
	| '.' Hexadecimal_digit_sequence
	| Hexadecimal_digit_sequence '.';

/* ?
fractional-constant:
	digit-sequenceopt . digit-sequence
	| digit-sequence .
 */

fragment Fractional_const:
	Digit_sequence '.' Digit_sequence
	| '.' Digit_sequence
	| Digit_sequence '.';

/*
hexadecimal-floating-constant:
	hexadecimal-prefix hexadecimal-fractional-constant
		binary-exponent-part floating-suffix_opt
	| hexadecimal-prefix hexadecimal-digit-sequence
		binary-exponent-part floating-suffix_opt
 */
fragment Hexadecimal_floating_const:
	Hexadecimal_prefix (
		Hexadecimal_fractional_const
		| Hexadecimal_digit_sequence
	) Binary_exponent_part;
/*
decimal-floating-constant:
	fractional-constant exponent-partopt floating-suffixopt
	| digit-sequence exponent-part floating-suffixopt
 */

fragment Decimal_floating_const:
	Fractional_const
	| (Fractional_const | Digit_sequence) Exponent_part;

FLOATING_CONST:
	Decimal_floating_const
	| Hexadecimal_floating_const;

// escape tab and enter
WS: [ \t\r\n] -> skip;

/*sysy-2022-spec-P4 2.comments*/
COMMENT: '/*' .*? '*/' -> skip;
LINE_COMMENT: '//' .*? '\r'? '\n' -> skip;

/*===-------------------------------------------===*/
/* Syntax rules                                    */
/*===-------------------------------------------===*/

/*Syntax Definitions sysy-spec-P2*/
// Compile Unit
/* CompUnit → [ CompUnit ] ( Decl | FuncDef ) */
compUnit: (decl | funcDef)+ EOF;

// Declaration
/* Decl → ConstDecl | VarDecl */
decl: constDecl | varDecl;

// const value declarations
/* ConstDecl → 'const' BType ConstDef { ',' ConstDef } ';' */
constDecl:
	CONSTLABEL bType constDef (COMMA constDef)* SEMICOLON;

// basic type
bType: INTTYPE | FLOATTYPE;

// definitions of const value
/* ConstDef → Ident { '[' ConstExp ']' } '=' ConstInitVal */
constDef:
	IDENTIFIER (L_BRACKET constExp R_BRACKET)* ASSIGNMARK constInitVal;

// inital value of const
/* 
ConstInitVal -> 
	ConstExp 
	| '{' [ ConstInitVal { ',' ConstInitVal } ] '}'
 */
constInitVal:
	constExp
	| L_BRACE (constInitVal (COMMA constInitVal)*)? R_BRACE;

// variable decl
/* VarDecl → BType VarDef { ',' VarDef } ';' */
varDecl: bType varDef (COMMA varDef)* SEMICOLON;

// variable definition
/*
VarDef ->
	Ident { '[' ConstExp ']' }
	| Ident { '[' ConstExp ']' } '=' InitVal
*/
varDef:
	IDENTIFIER (L_BRACKET constExp R_BRACKET)* (ASSIGNMARK initVal)?;

// initial value of variables
/* InitVal -> Exp | '{' [ InitVal { ',' InitVal } ] '}' */
initVal: exp | L_BRACE (initVal (COMMA initVal)*)? R_BRACE;

// definitions of function
/* FuncDef -> FuncType Ident '(' [FuncFParams] ')' Block */
funcDef:
	funcType IDENTIFIER L_PARENTHESIS (funcFParams)? R_PARENTHESIS block;

// types of functions
/* FuncType -> 'void' | 'int' | 'float' */
funcType: INTTYPE | FLOATTYPE | VOIDTYPE;

// formative parameters list of function
/* FuncFParams -> FuncFParam { ',' FuncFParam } */
funcFParams: funcFParam (COMMA funcFParam)*;

// formative parameter of function
/* FuncFParam -> BType Ident ['[' ']' { '[' Exp ']' }] */
funcFParam:
	bType IDENTIFIER (L_BRACKET R_BRACKET (L_BRACKET exp R_BRACKET)*)?;

// block of statements
/* Block -> '{' { BlockItem } '}' */
block: L_BRACE blockItem* R_BRACE;

// per item in a statement block
/* BlockItem -> Decl | Stmt */
blockItem: decl | stmt;

// def of statement
/* 
Stmt -> 
	  LVal '=' Exp ';' 
	| [Exp] ';' 
	| Block
	| 'if' '( Cond ')' Stmt [ 'else' Stmt ]
	| 'while' '(' Cond ')' Stmt
	| 'break' ';' | 'continue' ';'
	| 'return' [Exp] ';'
 */
stmt:
	lVal ASSIGNMARK exp SEMICOLON
	| exp? SEMICOLON
	| block
	| IFKEY L_PARENTHESIS cond R_PARENTHESIS stmt (ELSEKEY stmt)?
	| WHILEKEY L_PARENTHESIS cond R_PARENTHESIS stmt
	| BREAKKEY SEMICOLON
	| CONTINUEKEY SEMICOLON
	| RETURNKEY exp? SEMICOLON;

// expression tip: in sysY, expressions are all integer or floating
exp: addExp;

// condition expression
cond: lOrExp;

// expression of left value
/* LVal -> Ident {'[' Exp ']'} */
lVal: IDENTIFIER (L_BRACKET exp R_BRACKET)*;

// primary expressions
/* PrimaryExp -> '(' Exp ')' | LVal | Number */
primaryExp: L_PARENTHESIS exp R_PARENTHESIS | lVal | number;

//number
/* Number -> IntConst | floatConst */
number: INTEGER_CONST | FLOATING_CONST;

// unary expression
/* 
UnaryExp ->   PrimaryExp 
			| Ident '(' [FuncRParams] ')' 
			| UnaryOp UnaryEx
 */
unaryExp:
	primaryExp
	| IDENTIFIER L_PARENTHESIS funcRParams? R_PARENTHESIS
	| unaryOp unaryExp;

// unary operator
/* UnaryOp -> '+' | '−' | '!' */
unaryOp: ADDOP | MINUSOP | NOTOP;

// real parameters of function
/* FuncRParams -> Exp { ',' Exp } */
funcRParams: exp (COMMA exp)*;

// expression with * / %
/* MulExp -> 
	UnaryExp 
	| MulExp ('*' | '/' | '%') UnaryExp 
*/
mulExp: unaryExp | mulExp MULOP unaryExp;

// expression with + -
/* AddExp -> MulExp | AddExp ('+' | '−') MulExp */
addExp: mulExp | addExp (ADDOP | MINUSOP) mulExp;

// expression with relation operator
/* RelExp -> AddExp | RelExp ('<' | '>' | '<=' | '>=') AddExp */
relExp: addExp | relExp RELOP addExp;

// epression to judge equalty
/* EqExp -> RelExp | EqExp ('==' | '!=') RelExp */
eqExp: relExp | eqExp EQOP relExp;

// logic and expression
/* LAndExp -> EqExp | LAndExp '&&' EqExp */
lAndExp: eqExp | lAndExp LANDOP eqExp;

// logic or expression
/* LOrExp -> LAndExp | LOrExp '||' LAndExp */
lOrExp: lAndExp | lOrExp LOROP lAndExp;

// const expressions all identifiers must be constant
/* ConstExp -> AddExp  */
constExp: addExp;