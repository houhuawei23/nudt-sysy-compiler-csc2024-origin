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
    - s
    - get dims

- visitLocalDecl:
  - decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
  - vector<Value *> values
  - create pointer type point to btype, as alloca inst type?
    - type: PointerType(btype)
  - for each varDef:
    - `varDef: lValue (ASSIGN initValue)?;`
    - `lValue: ID (LBRACKET exp RBRACKET)*;`
    - get dimensions: dims
      - `a[2][2] = {1, 2, 3}`
      - `exp = [1, 2]`
      - for dim in exp:
        - cast dim to value
        - dims.push_back(dim)
        - `dims = vector({1, 2})` in value type
    - builder create alloca inst: alloca (type, dims, name, isconst)
      - [return] type: type ？
      - operands: dims
    - if has assign stmt:
      - if scalar:
        - create value by type
        - type cast if need
        - builder create store inst: store (value, alloca)
        - if const:
          - create const value by dynamic_cast
          - set alloca instruction type by const type
      - if array:
        - d = 0, n = 0
        - path = vector<int>(dims, 0)
        - isalloca = true
        - current_alloca = alloca
        - get base type?
          - current_type = alloca->getType()->as<PointerType>()->getBaseType() ?
        - numdims = alloca->getNumDims() // int
        - for each initValue in varDef:
          - visitInitValue(init)
    - values.push_back(alloca)
  - return values

- visitInitValue
  - initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
  - by the father node decl:lValue = initVAlue;
    - have lValue current_type, numdims
    - have lValue numdims
    - path = vector<int>(alloca->getNumDims(), 0);
    - n = 0, d = 0
    - isalloca = true
    - current_alloca: new alloca inst
    -
  - if (ctx->exp()) : not the array??
    - visit(exp) to get the value
    - if is constant value:
      - get the (int|float) value by type
    - else if (l is int and r is float):
      - create float to int Instruction
    - else if (l is float and r is int):
      - create int to float Instruction
    - goto the last dimension
      - ??

  - else: array
    - visit(initValue) to get the array
    - 

- AllocaInst(type, dims, name, isconst)

- visitLValueExp
  - lValue: ID (LBRACKET exp RBRACKET)*;

- User: Value
  - vector<Use> operands; // 操作数 vector

- Instruction: User
  - parent -> parent basic block
  - protect_offset
  - pass_offset
  - protect_cnt
    std::set<Instruction *> front_live; // 该指令前面点的活跃变量
    std::set<Instruction*> back_live;  // 该指令后面点的活跃变量
    std::set<Value *> front_vlive;      // 该指令前面点的活跃变量
    std::set<Value*> back_vlive;       // 该指令后面点的活跃变量
