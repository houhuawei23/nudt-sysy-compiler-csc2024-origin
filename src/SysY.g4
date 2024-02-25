/*===-------------------------------------------===*/
/* SysY.g4 CREATED BY TXS 2024-2-16                */
/*===-------------------------------------------===*/
grammar SysY;
import SysYLexerRules;
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