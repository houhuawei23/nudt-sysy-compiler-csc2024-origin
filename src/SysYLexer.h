
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
    ADDOP = 20, MINUSOP = 21, NOTOP = 22, MULOP = 23, RELOP = 24, EQOP = 25, 
    LANDOP = 26, LOROP = 27, IDENTIFIER = 28, INTEGER_CONST = 29, FLOATING_CONST = 30, 
    WS = 31, COMMENT = 32, LINE_COMMENT = 33
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

