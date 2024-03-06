# Visit

- visitFunc:
  - func: funcType ID LPAREN funcFParams? RPAREN blockStmt;
  - Function:
    - parent_module: father module the func belong to
    - blocks: basic blocks belong to the func
    - Construct: init, add entry block as first block
    - add_bblock
  - get name by ID
  - parse funcFParams to get paramTypes, paramNames
  - get return type by funcType: INT or FLOAT
  - get function type by (returnType, paramTypes)
  - create funtion and update the symbol table
  - create function scope / create new local symbol table for func
  - get entry block and insert the params? create argument?
  - ? builder: setup the insrtuction insert point
  - visit block statements - generate function body
  - return function

- visitBlockStmt:
  - blockStmt: LBRACE blockItem* RBRACE;
  - return builder.get_basic_block ?

- visitBlockItem:
  - blockItem: decl | stmt;
  - default: visitChildren(ctx)

- visitDecl:
  - decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
  - is module scope / is global decl?
    - visitGlobalDecl
    - visitLocalDecl

- visitGlobalDecl
  - decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
  - isConst?
  - get type by btype
  - for each varDef:
    - varDef: lValue (ASSIGN initValue)?;
    - 
    - get dims
