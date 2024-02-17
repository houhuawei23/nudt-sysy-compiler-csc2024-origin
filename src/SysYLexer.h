
// Generated from SysY.g4 by ANTLR 4.12.0

#pragma once


#include "antlr4-runtime.h"




class  SysYLexer : public antlr4::Lexer {
public:
  enum {
    COMMA = 1, CONSTLABEL = 2, INTTYPE = 3, FLOATTYPE = 4, SEMICOLON = 5, 
    L_PARENTHESIS = 6, R_PARENTHESIS = 7, L_BRACKET = 8, R_BRACKET = 9, 
    L_BRACE = 10, R_BRACE = 11, ASSIGNMARK = 12, VOIDTYPE = 13, IFKEY = 14, 
    ELSEKEY = 15, WHILEKEY = 16, BREAKKEY = 17, CONTINUEKEY = 18, RETURNKEY = 19, 
    UNARYOP = 20, MULOP = 21, ADDOP = 22, RELOP = 23, EQOP = 24, LANDOP = 25, 
    LOROP = 26, IDENTIFIER = 27, INTEGER_CONST = 28, FLOATING_CONST = 29, 
    WS = 30, COMMENT = 31, LINE_COMMENT = 32
  };

  explicit SysYLexer(antlr4::CharStream *input);

  ~SysYLexer() override;


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};

