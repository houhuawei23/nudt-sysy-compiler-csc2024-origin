/*===-------------------------------------------===*/
/*    SysYLexerRules.g4 CREATED BY HHW 2024-2-24   */
/*         Auxillary rules and Lexer rules         */
/*===-------------------------------------------===*/
lexer grammar SysYLexerRules;

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
	digit-sequence_opt . digit-sequence
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