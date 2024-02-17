
// Generated from SysY.g4 by ANTLR 4.12.0


#include "SysYVisitor.h"

#include "SysYParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct SysYParserStaticData final {
  SysYParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  SysYParserStaticData(const SysYParserStaticData&) = delete;
  SysYParserStaticData(SysYParserStaticData&&) = delete;
  SysYParserStaticData& operator=(const SysYParserStaticData&) = delete;
  SysYParserStaticData& operator=(SysYParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag sysyParserOnceFlag;
SysYParserStaticData *sysyParserStaticData = nullptr;

void sysyParserInitialize() {
  assert(sysyParserStaticData == nullptr);
  auto staticData = std::make_unique<SysYParserStaticData>(
    std::vector<std::string>{
      "compUnit", "decl", "constDecl", "bType", "constDef", "constInitVal", 
      "varDecl", "varDef", "initVal", "funcDef", "funcType", "funcFParams", 
      "funcFParam", "block", "blockItem", "stmt", "exp", "cond", "lVal", 
      "primaryExp", "number", "unaryExp", "unaryOp", "funcRParams", "mulExp", 
      "addExp", "relExp", "eqExp", "lAndExp", "lOrExp", "constExp"
    },
    std::vector<std::string>{
      "", "','", "'const'", "'int'", "'float'", "';'", "'('", "')'", "'['", 
      "']'", "'{'", "'}'", "'='", "'void'", "'if'", "'else'", "'while'", 
      "'break'", "'continue'", "'return'", "", "", "", "", "", "'&&'", "'||'"
    },
    std::vector<std::string>{
      "", "COMMA", "CONSTLABEL", "INTTYPE", "FLOATTYPE", "SEMICOLON", "L_PARENTHESIS", 
      "R_PARENTHESIS", "L_BRACKET", "R_BRACKET", "L_BRACE", "R_BRACE", "ASSIGNMARK", 
      "VOIDTYPE", "IFKEY", "ELSEKEY", "WHILEKEY", "BREAKKEY", "CONTINUEKEY", 
      "RETURNKEY", "UNARYOP", "MULOP", "ADDOP", "RELOP", "EQOP", "LANDOP", 
      "LOROP", "IDENTIFIER", "INTEGER_CONST", "FLOATING_CONST", "WS", "COMMENT", 
      "LINE_COMMENT"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,32,355,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,2,14,7,
  	14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,7,20,2,21,7,
  	21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,2,27,7,27,2,28,7,
  	28,2,29,7,29,2,30,7,30,1,0,1,0,5,0,65,8,0,10,0,12,0,68,9,0,1,0,1,0,1,
  	1,1,1,3,1,74,8,1,1,2,1,2,1,2,1,2,1,2,5,2,81,8,2,10,2,12,2,84,9,2,1,2,
  	1,2,1,3,1,3,1,4,1,4,1,4,1,4,1,4,5,4,95,8,4,10,4,12,4,98,9,4,1,4,1,4,1,
  	4,1,5,1,5,1,5,1,5,1,5,5,5,108,8,5,10,5,12,5,111,9,5,3,5,113,8,5,1,5,3,
  	5,116,8,5,1,6,1,6,1,6,1,6,5,6,122,8,6,10,6,12,6,125,9,6,1,6,1,6,1,7,1,
  	7,1,7,1,7,1,7,5,7,134,8,7,10,7,12,7,137,9,7,1,7,1,7,3,7,141,8,7,1,8,1,
  	8,1,8,1,8,1,8,5,8,148,8,8,10,8,12,8,151,9,8,3,8,153,8,8,1,8,3,8,156,8,
  	8,1,9,1,9,1,9,1,9,3,9,162,8,9,1,9,1,9,1,9,1,10,1,10,1,11,1,11,1,11,5,
  	11,172,8,11,10,11,12,11,175,9,11,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,
  	12,5,12,185,8,12,10,12,12,12,188,9,12,3,12,190,8,12,1,13,1,13,5,13,194,
  	8,13,10,13,12,13,197,9,13,1,13,1,13,1,14,1,14,3,14,203,8,14,1,15,1,15,
  	1,15,1,15,1,15,1,15,3,15,211,8,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,
  	1,15,1,15,3,15,222,8,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,
  	1,15,1,15,1,15,3,15,236,8,15,1,15,3,15,239,8,15,1,16,1,16,1,17,1,17,1,
  	18,1,18,1,18,1,18,1,18,5,18,250,8,18,10,18,12,18,253,9,18,1,19,1,19,1,
  	19,1,19,1,19,1,19,3,19,261,8,19,1,20,1,20,1,21,1,21,1,21,1,21,3,21,269,
  	8,21,1,21,1,21,1,21,1,21,3,21,275,8,21,1,22,1,22,1,23,1,23,1,23,5,23,
  	282,8,23,10,23,12,23,285,9,23,1,24,1,24,1,24,1,24,1,24,1,24,5,24,293,
  	8,24,10,24,12,24,296,9,24,1,25,1,25,1,25,1,25,1,25,1,25,5,25,304,8,25,
  	10,25,12,25,307,9,25,1,26,1,26,1,26,1,26,1,26,1,26,5,26,315,8,26,10,26,
  	12,26,318,9,26,1,27,1,27,1,27,1,27,1,27,1,27,5,27,326,8,27,10,27,12,27,
  	329,9,27,1,28,1,28,1,28,1,28,1,28,1,28,5,28,337,8,28,10,28,12,28,340,
  	9,28,1,29,1,29,1,29,1,29,1,29,1,29,5,29,348,8,29,10,29,12,29,351,9,29,
  	1,30,1,30,1,30,0,6,48,50,52,54,56,58,31,0,2,4,6,8,10,12,14,16,18,20,22,
  	24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,0,3,1,0,3,4,
  	2,0,3,4,13,13,1,0,28,29,366,0,66,1,0,0,0,2,73,1,0,0,0,4,75,1,0,0,0,6,
  	87,1,0,0,0,8,89,1,0,0,0,10,115,1,0,0,0,12,117,1,0,0,0,14,128,1,0,0,0,
  	16,155,1,0,0,0,18,157,1,0,0,0,20,166,1,0,0,0,22,168,1,0,0,0,24,176,1,
  	0,0,0,26,191,1,0,0,0,28,202,1,0,0,0,30,238,1,0,0,0,32,240,1,0,0,0,34,
  	242,1,0,0,0,36,244,1,0,0,0,38,260,1,0,0,0,40,262,1,0,0,0,42,274,1,0,0,
  	0,44,276,1,0,0,0,46,278,1,0,0,0,48,286,1,0,0,0,50,297,1,0,0,0,52,308,
  	1,0,0,0,54,319,1,0,0,0,56,330,1,0,0,0,58,341,1,0,0,0,60,352,1,0,0,0,62,
  	65,3,2,1,0,63,65,3,18,9,0,64,62,1,0,0,0,64,63,1,0,0,0,65,68,1,0,0,0,66,
  	64,1,0,0,0,66,67,1,0,0,0,67,69,1,0,0,0,68,66,1,0,0,0,69,70,5,0,0,1,70,
  	1,1,0,0,0,71,74,3,4,2,0,72,74,3,12,6,0,73,71,1,0,0,0,73,72,1,0,0,0,74,
  	3,1,0,0,0,75,76,5,2,0,0,76,77,3,6,3,0,77,82,3,8,4,0,78,79,5,1,0,0,79,
  	81,3,8,4,0,80,78,1,0,0,0,81,84,1,0,0,0,82,80,1,0,0,0,82,83,1,0,0,0,83,
  	85,1,0,0,0,84,82,1,0,0,0,85,86,5,5,0,0,86,5,1,0,0,0,87,88,7,0,0,0,88,
  	7,1,0,0,0,89,96,5,27,0,0,90,91,5,8,0,0,91,92,3,60,30,0,92,93,5,9,0,0,
  	93,95,1,0,0,0,94,90,1,0,0,0,95,98,1,0,0,0,96,94,1,0,0,0,96,97,1,0,0,0,
  	97,99,1,0,0,0,98,96,1,0,0,0,99,100,5,12,0,0,100,101,3,10,5,0,101,9,1,
  	0,0,0,102,116,3,60,30,0,103,112,5,10,0,0,104,109,3,10,5,0,105,106,5,1,
  	0,0,106,108,3,10,5,0,107,105,1,0,0,0,108,111,1,0,0,0,109,107,1,0,0,0,
  	109,110,1,0,0,0,110,113,1,0,0,0,111,109,1,0,0,0,112,104,1,0,0,0,112,113,
  	1,0,0,0,113,114,1,0,0,0,114,116,5,11,0,0,115,102,1,0,0,0,115,103,1,0,
  	0,0,116,11,1,0,0,0,117,118,3,6,3,0,118,123,3,14,7,0,119,120,5,1,0,0,120,
  	122,3,14,7,0,121,119,1,0,0,0,122,125,1,0,0,0,123,121,1,0,0,0,123,124,
  	1,0,0,0,124,126,1,0,0,0,125,123,1,0,0,0,126,127,5,5,0,0,127,13,1,0,0,
  	0,128,135,5,27,0,0,129,130,5,8,0,0,130,131,3,60,30,0,131,132,5,9,0,0,
  	132,134,1,0,0,0,133,129,1,0,0,0,134,137,1,0,0,0,135,133,1,0,0,0,135,136,
  	1,0,0,0,136,140,1,0,0,0,137,135,1,0,0,0,138,139,5,12,0,0,139,141,3,16,
  	8,0,140,138,1,0,0,0,140,141,1,0,0,0,141,15,1,0,0,0,142,156,3,32,16,0,
  	143,152,5,10,0,0,144,149,3,16,8,0,145,146,5,1,0,0,146,148,3,16,8,0,147,
  	145,1,0,0,0,148,151,1,0,0,0,149,147,1,0,0,0,149,150,1,0,0,0,150,153,1,
  	0,0,0,151,149,1,0,0,0,152,144,1,0,0,0,152,153,1,0,0,0,153,154,1,0,0,0,
  	154,156,5,11,0,0,155,142,1,0,0,0,155,143,1,0,0,0,156,17,1,0,0,0,157,158,
  	3,20,10,0,158,159,5,27,0,0,159,161,5,6,0,0,160,162,3,22,11,0,161,160,
  	1,0,0,0,161,162,1,0,0,0,162,163,1,0,0,0,163,164,5,7,0,0,164,165,3,26,
  	13,0,165,19,1,0,0,0,166,167,7,1,0,0,167,21,1,0,0,0,168,173,3,24,12,0,
  	169,170,5,1,0,0,170,172,3,24,12,0,171,169,1,0,0,0,172,175,1,0,0,0,173,
  	171,1,0,0,0,173,174,1,0,0,0,174,23,1,0,0,0,175,173,1,0,0,0,176,177,3,
  	6,3,0,177,189,5,27,0,0,178,179,5,8,0,0,179,186,5,9,0,0,180,181,5,8,0,
  	0,181,182,3,32,16,0,182,183,5,9,0,0,183,185,1,0,0,0,184,180,1,0,0,0,185,
  	188,1,0,0,0,186,184,1,0,0,0,186,187,1,0,0,0,187,190,1,0,0,0,188,186,1,
  	0,0,0,189,178,1,0,0,0,189,190,1,0,0,0,190,25,1,0,0,0,191,195,5,10,0,0,
  	192,194,3,28,14,0,193,192,1,0,0,0,194,197,1,0,0,0,195,193,1,0,0,0,195,
  	196,1,0,0,0,196,198,1,0,0,0,197,195,1,0,0,0,198,199,5,11,0,0,199,27,1,
  	0,0,0,200,203,3,2,1,0,201,203,3,30,15,0,202,200,1,0,0,0,202,201,1,0,0,
  	0,203,29,1,0,0,0,204,205,3,36,18,0,205,206,5,12,0,0,206,207,3,32,16,0,
  	207,208,5,5,0,0,208,239,1,0,0,0,209,211,3,32,16,0,210,209,1,0,0,0,210,
  	211,1,0,0,0,211,212,1,0,0,0,212,239,5,5,0,0,213,239,3,26,13,0,214,215,
  	5,14,0,0,215,216,5,6,0,0,216,217,3,34,17,0,217,218,5,7,0,0,218,221,3,
  	30,15,0,219,220,5,15,0,0,220,222,3,30,15,0,221,219,1,0,0,0,221,222,1,
  	0,0,0,222,239,1,0,0,0,223,224,5,16,0,0,224,225,5,6,0,0,225,226,3,34,17,
  	0,226,227,5,7,0,0,227,228,3,30,15,0,228,239,1,0,0,0,229,230,5,17,0,0,
  	230,239,5,5,0,0,231,232,5,18,0,0,232,239,5,5,0,0,233,235,5,19,0,0,234,
  	236,3,32,16,0,235,234,1,0,0,0,235,236,1,0,0,0,236,237,1,0,0,0,237,239,
  	5,5,0,0,238,204,1,0,0,0,238,210,1,0,0,0,238,213,1,0,0,0,238,214,1,0,0,
  	0,238,223,1,0,0,0,238,229,1,0,0,0,238,231,1,0,0,0,238,233,1,0,0,0,239,
  	31,1,0,0,0,240,241,3,50,25,0,241,33,1,0,0,0,242,243,3,58,29,0,243,35,
  	1,0,0,0,244,251,5,27,0,0,245,246,5,8,0,0,246,247,3,32,16,0,247,248,5,
  	9,0,0,248,250,1,0,0,0,249,245,1,0,0,0,250,253,1,0,0,0,251,249,1,0,0,0,
  	251,252,1,0,0,0,252,37,1,0,0,0,253,251,1,0,0,0,254,255,5,6,0,0,255,256,
  	3,32,16,0,256,257,5,7,0,0,257,261,1,0,0,0,258,261,3,36,18,0,259,261,3,
  	40,20,0,260,254,1,0,0,0,260,258,1,0,0,0,260,259,1,0,0,0,261,39,1,0,0,
  	0,262,263,7,2,0,0,263,41,1,0,0,0,264,275,3,38,19,0,265,266,5,27,0,0,266,
  	268,5,6,0,0,267,269,3,22,11,0,268,267,1,0,0,0,268,269,1,0,0,0,269,270,
  	1,0,0,0,270,275,5,7,0,0,271,272,3,44,22,0,272,273,3,42,21,0,273,275,1,
  	0,0,0,274,264,1,0,0,0,274,265,1,0,0,0,274,271,1,0,0,0,275,43,1,0,0,0,
  	276,277,5,20,0,0,277,45,1,0,0,0,278,283,3,32,16,0,279,280,5,1,0,0,280,
  	282,3,32,16,0,281,279,1,0,0,0,282,285,1,0,0,0,283,281,1,0,0,0,283,284,
  	1,0,0,0,284,47,1,0,0,0,285,283,1,0,0,0,286,287,6,24,-1,0,287,288,3,42,
  	21,0,288,294,1,0,0,0,289,290,10,1,0,0,290,291,5,21,0,0,291,293,3,42,21,
  	0,292,289,1,0,0,0,293,296,1,0,0,0,294,292,1,0,0,0,294,295,1,0,0,0,295,
  	49,1,0,0,0,296,294,1,0,0,0,297,298,6,25,-1,0,298,299,3,48,24,0,299,305,
  	1,0,0,0,300,301,10,1,0,0,301,302,5,22,0,0,302,304,3,48,24,0,303,300,1,
  	0,0,0,304,307,1,0,0,0,305,303,1,0,0,0,305,306,1,0,0,0,306,51,1,0,0,0,
  	307,305,1,0,0,0,308,309,6,26,-1,0,309,310,3,50,25,0,310,316,1,0,0,0,311,
  	312,10,1,0,0,312,313,5,23,0,0,313,315,3,50,25,0,314,311,1,0,0,0,315,318,
  	1,0,0,0,316,314,1,0,0,0,316,317,1,0,0,0,317,53,1,0,0,0,318,316,1,0,0,
  	0,319,320,6,27,-1,0,320,321,3,52,26,0,321,327,1,0,0,0,322,323,10,1,0,
  	0,323,324,5,24,0,0,324,326,3,52,26,0,325,322,1,0,0,0,326,329,1,0,0,0,
  	327,325,1,0,0,0,327,328,1,0,0,0,328,55,1,0,0,0,329,327,1,0,0,0,330,331,
  	6,28,-1,0,331,332,3,54,27,0,332,338,1,0,0,0,333,334,10,1,0,0,334,335,
  	5,25,0,0,335,337,3,54,27,0,336,333,1,0,0,0,337,340,1,0,0,0,338,336,1,
  	0,0,0,338,339,1,0,0,0,339,57,1,0,0,0,340,338,1,0,0,0,341,342,6,29,-1,
  	0,342,343,3,56,28,0,343,349,1,0,0,0,344,345,10,1,0,0,345,346,5,26,0,0,
  	346,348,3,56,28,0,347,344,1,0,0,0,348,351,1,0,0,0,349,347,1,0,0,0,349,
  	350,1,0,0,0,350,59,1,0,0,0,351,349,1,0,0,0,352,353,3,50,25,0,353,61,1,
  	0,0,0,35,64,66,73,82,96,109,112,115,123,135,140,149,152,155,161,173,186,
  	189,195,202,210,221,235,238,251,260,268,274,283,294,305,316,327,338,349
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  sysyParserStaticData = staticData.release();
}

}

SysYParser::SysYParser(TokenStream *input) : SysYParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

SysYParser::SysYParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  SysYParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *sysyParserStaticData->atn, sysyParserStaticData->decisionToDFA, sysyParserStaticData->sharedContextCache, options);
}

SysYParser::~SysYParser() {
  delete _interpreter;
}

const atn::ATN& SysYParser::getATN() const {
  return *sysyParserStaticData->atn;
}

std::string SysYParser::getGrammarFileName() const {
  return "SysY.g4";
}

const std::vector<std::string>& SysYParser::getRuleNames() const {
  return sysyParserStaticData->ruleNames;
}

const dfa::Vocabulary& SysYParser::getVocabulary() const {
  return sysyParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView SysYParser::getSerializedATN() const {
  return sysyParserStaticData->serializedATN;
}


//----------------- CompUnitContext ------------------------------------------------------------------

SysYParser::CompUnitContext::CompUnitContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::CompUnitContext::EOF() {
  return getToken(SysYParser::EOF, 0);
}

std::vector<SysYParser::DeclContext *> SysYParser::CompUnitContext::decl() {
  return getRuleContexts<SysYParser::DeclContext>();
}

SysYParser::DeclContext* SysYParser::CompUnitContext::decl(size_t i) {
  return getRuleContext<SysYParser::DeclContext>(i);
}

std::vector<SysYParser::FuncDefContext *> SysYParser::CompUnitContext::funcDef() {
  return getRuleContexts<SysYParser::FuncDefContext>();
}

SysYParser::FuncDefContext* SysYParser::CompUnitContext::funcDef(size_t i) {
  return getRuleContext<SysYParser::FuncDefContext>(i);
}


size_t SysYParser::CompUnitContext::getRuleIndex() const {
  return SysYParser::RuleCompUnit;
}


std::any SysYParser::CompUnitContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitCompUnit(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::CompUnitContext* SysYParser::compUnit() {
  CompUnitContext *_localctx = _tracker.createInstance<CompUnitContext>(_ctx, getState());
  enterRule(_localctx, 0, SysYParser::RuleCompUnit);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(66);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 8220) != 0)) {
      setState(64);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx)) {
      case 1: {
        setState(62);
        decl();
        break;
      }

      case 2: {
        setState(63);
        funcDef();
        break;
      }

      default:
        break;
      }
      setState(68);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(69);
    match(SysYParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DeclContext ------------------------------------------------------------------

SysYParser::DeclContext::DeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::ConstDeclContext* SysYParser::DeclContext::constDecl() {
  return getRuleContext<SysYParser::ConstDeclContext>(0);
}

SysYParser::VarDeclContext* SysYParser::DeclContext::varDecl() {
  return getRuleContext<SysYParser::VarDeclContext>(0);
}


size_t SysYParser::DeclContext::getRuleIndex() const {
  return SysYParser::RuleDecl;
}


std::any SysYParser::DeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitDecl(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::DeclContext* SysYParser::decl() {
  DeclContext *_localctx = _tracker.createInstance<DeclContext>(_ctx, getState());
  enterRule(_localctx, 2, SysYParser::RuleDecl);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(73);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SysYParser::CONSTLABEL: {
        enterOuterAlt(_localctx, 1);
        setState(71);
        constDecl();
        break;
      }

      case SysYParser::INTTYPE:
      case SysYParser::FLOATTYPE: {
        enterOuterAlt(_localctx, 2);
        setState(72);
        varDecl();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConstDeclContext ------------------------------------------------------------------

SysYParser::ConstDeclContext::ConstDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::ConstDeclContext::CONSTLABEL() {
  return getToken(SysYParser::CONSTLABEL, 0);
}

SysYParser::BTypeContext* SysYParser::ConstDeclContext::bType() {
  return getRuleContext<SysYParser::BTypeContext>(0);
}

std::vector<SysYParser::ConstDefContext *> SysYParser::ConstDeclContext::constDef() {
  return getRuleContexts<SysYParser::ConstDefContext>();
}

SysYParser::ConstDefContext* SysYParser::ConstDeclContext::constDef(size_t i) {
  return getRuleContext<SysYParser::ConstDefContext>(i);
}

tree::TerminalNode* SysYParser::ConstDeclContext::SEMICOLON() {
  return getToken(SysYParser::SEMICOLON, 0);
}

std::vector<tree::TerminalNode *> SysYParser::ConstDeclContext::COMMA() {
  return getTokens(SysYParser::COMMA);
}

tree::TerminalNode* SysYParser::ConstDeclContext::COMMA(size_t i) {
  return getToken(SysYParser::COMMA, i);
}


size_t SysYParser::ConstDeclContext::getRuleIndex() const {
  return SysYParser::RuleConstDecl;
}


std::any SysYParser::ConstDeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitConstDecl(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::ConstDeclContext* SysYParser::constDecl() {
  ConstDeclContext *_localctx = _tracker.createInstance<ConstDeclContext>(_ctx, getState());
  enterRule(_localctx, 4, SysYParser::RuleConstDecl);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(75);
    match(SysYParser::CONSTLABEL);
    setState(76);
    bType();
    setState(77);
    constDef();
    setState(82);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SysYParser::COMMA) {
      setState(78);
      match(SysYParser::COMMA);
      setState(79);
      constDef();
      setState(84);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(85);
    match(SysYParser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BTypeContext ------------------------------------------------------------------

SysYParser::BTypeContext::BTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::BTypeContext::INTTYPE() {
  return getToken(SysYParser::INTTYPE, 0);
}

tree::TerminalNode* SysYParser::BTypeContext::FLOATTYPE() {
  return getToken(SysYParser::FLOATTYPE, 0);
}


size_t SysYParser::BTypeContext::getRuleIndex() const {
  return SysYParser::RuleBType;
}


std::any SysYParser::BTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitBType(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::BTypeContext* SysYParser::bType() {
  BTypeContext *_localctx = _tracker.createInstance<BTypeContext>(_ctx, getState());
  enterRule(_localctx, 6, SysYParser::RuleBType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(87);
    _la = _input->LA(1);
    if (!(_la == SysYParser::INTTYPE

    || _la == SysYParser::FLOATTYPE)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConstDefContext ------------------------------------------------------------------

SysYParser::ConstDefContext::ConstDefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::ConstDefContext::IDENTIFIER() {
  return getToken(SysYParser::IDENTIFIER, 0);
}

tree::TerminalNode* SysYParser::ConstDefContext::ASSIGNMARK() {
  return getToken(SysYParser::ASSIGNMARK, 0);
}

SysYParser::ConstInitValContext* SysYParser::ConstDefContext::constInitVal() {
  return getRuleContext<SysYParser::ConstInitValContext>(0);
}

std::vector<tree::TerminalNode *> SysYParser::ConstDefContext::L_BRACKET() {
  return getTokens(SysYParser::L_BRACKET);
}

tree::TerminalNode* SysYParser::ConstDefContext::L_BRACKET(size_t i) {
  return getToken(SysYParser::L_BRACKET, i);
}

std::vector<SysYParser::ConstExpContext *> SysYParser::ConstDefContext::constExp() {
  return getRuleContexts<SysYParser::ConstExpContext>();
}

SysYParser::ConstExpContext* SysYParser::ConstDefContext::constExp(size_t i) {
  return getRuleContext<SysYParser::ConstExpContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::ConstDefContext::R_BRACKET() {
  return getTokens(SysYParser::R_BRACKET);
}

tree::TerminalNode* SysYParser::ConstDefContext::R_BRACKET(size_t i) {
  return getToken(SysYParser::R_BRACKET, i);
}


size_t SysYParser::ConstDefContext::getRuleIndex() const {
  return SysYParser::RuleConstDef;
}


std::any SysYParser::ConstDefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitConstDef(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::ConstDefContext* SysYParser::constDef() {
  ConstDefContext *_localctx = _tracker.createInstance<ConstDefContext>(_ctx, getState());
  enterRule(_localctx, 8, SysYParser::RuleConstDef);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(89);
    match(SysYParser::IDENTIFIER);
    setState(96);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SysYParser::L_BRACKET) {
      setState(90);
      match(SysYParser::L_BRACKET);
      setState(91);
      constExp();
      setState(92);
      match(SysYParser::R_BRACKET);
      setState(98);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(99);
    match(SysYParser::ASSIGNMARK);
    setState(100);
    constInitVal();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConstInitValContext ------------------------------------------------------------------

SysYParser::ConstInitValContext::ConstInitValContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::ConstExpContext* SysYParser::ConstInitValContext::constExp() {
  return getRuleContext<SysYParser::ConstExpContext>(0);
}

tree::TerminalNode* SysYParser::ConstInitValContext::L_BRACE() {
  return getToken(SysYParser::L_BRACE, 0);
}

tree::TerminalNode* SysYParser::ConstInitValContext::R_BRACE() {
  return getToken(SysYParser::R_BRACE, 0);
}

std::vector<SysYParser::ConstInitValContext *> SysYParser::ConstInitValContext::constInitVal() {
  return getRuleContexts<SysYParser::ConstInitValContext>();
}

SysYParser::ConstInitValContext* SysYParser::ConstInitValContext::constInitVal(size_t i) {
  return getRuleContext<SysYParser::ConstInitValContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::ConstInitValContext::COMMA() {
  return getTokens(SysYParser::COMMA);
}

tree::TerminalNode* SysYParser::ConstInitValContext::COMMA(size_t i) {
  return getToken(SysYParser::COMMA, i);
}


size_t SysYParser::ConstInitValContext::getRuleIndex() const {
  return SysYParser::RuleConstInitVal;
}


std::any SysYParser::ConstInitValContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitConstInitVal(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::ConstInitValContext* SysYParser::constInitVal() {
  ConstInitValContext *_localctx = _tracker.createInstance<ConstInitValContext>(_ctx, getState());
  enterRule(_localctx, 10, SysYParser::RuleConstInitVal);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(115);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SysYParser::L_PARENTHESIS:
      case SysYParser::UNARYOP:
      case SysYParser::IDENTIFIER:
      case SysYParser::INTEGER_CONST:
      case SysYParser::FLOATING_CONST: {
        enterOuterAlt(_localctx, 1);
        setState(102);
        constExp();
        break;
      }

      case SysYParser::L_BRACE: {
        enterOuterAlt(_localctx, 2);
        setState(103);
        match(SysYParser::L_BRACE);
        setState(112);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & 940573760) != 0)) {
          setState(104);
          constInitVal();
          setState(109);
          _errHandler->sync(this);
          _la = _input->LA(1);
          while (_la == SysYParser::COMMA) {
            setState(105);
            match(SysYParser::COMMA);
            setState(106);
            constInitVal();
            setState(111);
            _errHandler->sync(this);
            _la = _input->LA(1);
          }
        }
        setState(114);
        match(SysYParser::R_BRACE);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarDeclContext ------------------------------------------------------------------

SysYParser::VarDeclContext::VarDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::BTypeContext* SysYParser::VarDeclContext::bType() {
  return getRuleContext<SysYParser::BTypeContext>(0);
}

std::vector<SysYParser::VarDefContext *> SysYParser::VarDeclContext::varDef() {
  return getRuleContexts<SysYParser::VarDefContext>();
}

SysYParser::VarDefContext* SysYParser::VarDeclContext::varDef(size_t i) {
  return getRuleContext<SysYParser::VarDefContext>(i);
}

tree::TerminalNode* SysYParser::VarDeclContext::SEMICOLON() {
  return getToken(SysYParser::SEMICOLON, 0);
}

std::vector<tree::TerminalNode *> SysYParser::VarDeclContext::COMMA() {
  return getTokens(SysYParser::COMMA);
}

tree::TerminalNode* SysYParser::VarDeclContext::COMMA(size_t i) {
  return getToken(SysYParser::COMMA, i);
}


size_t SysYParser::VarDeclContext::getRuleIndex() const {
  return SysYParser::RuleVarDecl;
}


std::any SysYParser::VarDeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitVarDecl(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::VarDeclContext* SysYParser::varDecl() {
  VarDeclContext *_localctx = _tracker.createInstance<VarDeclContext>(_ctx, getState());
  enterRule(_localctx, 12, SysYParser::RuleVarDecl);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(117);
    bType();
    setState(118);
    varDef();
    setState(123);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SysYParser::COMMA) {
      setState(119);
      match(SysYParser::COMMA);
      setState(120);
      varDef();
      setState(125);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(126);
    match(SysYParser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarDefContext ------------------------------------------------------------------

SysYParser::VarDefContext::VarDefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::VarDefContext::IDENTIFIER() {
  return getToken(SysYParser::IDENTIFIER, 0);
}

std::vector<tree::TerminalNode *> SysYParser::VarDefContext::L_BRACKET() {
  return getTokens(SysYParser::L_BRACKET);
}

tree::TerminalNode* SysYParser::VarDefContext::L_BRACKET(size_t i) {
  return getToken(SysYParser::L_BRACKET, i);
}

std::vector<SysYParser::ConstExpContext *> SysYParser::VarDefContext::constExp() {
  return getRuleContexts<SysYParser::ConstExpContext>();
}

SysYParser::ConstExpContext* SysYParser::VarDefContext::constExp(size_t i) {
  return getRuleContext<SysYParser::ConstExpContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::VarDefContext::R_BRACKET() {
  return getTokens(SysYParser::R_BRACKET);
}

tree::TerminalNode* SysYParser::VarDefContext::R_BRACKET(size_t i) {
  return getToken(SysYParser::R_BRACKET, i);
}

tree::TerminalNode* SysYParser::VarDefContext::ASSIGNMARK() {
  return getToken(SysYParser::ASSIGNMARK, 0);
}

SysYParser::InitValContext* SysYParser::VarDefContext::initVal() {
  return getRuleContext<SysYParser::InitValContext>(0);
}


size_t SysYParser::VarDefContext::getRuleIndex() const {
  return SysYParser::RuleVarDef;
}


std::any SysYParser::VarDefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitVarDef(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::VarDefContext* SysYParser::varDef() {
  VarDefContext *_localctx = _tracker.createInstance<VarDefContext>(_ctx, getState());
  enterRule(_localctx, 14, SysYParser::RuleVarDef);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(128);
    match(SysYParser::IDENTIFIER);
    setState(135);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SysYParser::L_BRACKET) {
      setState(129);
      match(SysYParser::L_BRACKET);
      setState(130);
      constExp();
      setState(131);
      match(SysYParser::R_BRACKET);
      setState(137);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(140);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SysYParser::ASSIGNMARK) {
      setState(138);
      match(SysYParser::ASSIGNMARK);
      setState(139);
      initVal();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- InitValContext ------------------------------------------------------------------

SysYParser::InitValContext::InitValContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::ExpContext* SysYParser::InitValContext::exp() {
  return getRuleContext<SysYParser::ExpContext>(0);
}

tree::TerminalNode* SysYParser::InitValContext::L_BRACE() {
  return getToken(SysYParser::L_BRACE, 0);
}

tree::TerminalNode* SysYParser::InitValContext::R_BRACE() {
  return getToken(SysYParser::R_BRACE, 0);
}

std::vector<SysYParser::InitValContext *> SysYParser::InitValContext::initVal() {
  return getRuleContexts<SysYParser::InitValContext>();
}

SysYParser::InitValContext* SysYParser::InitValContext::initVal(size_t i) {
  return getRuleContext<SysYParser::InitValContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::InitValContext::COMMA() {
  return getTokens(SysYParser::COMMA);
}

tree::TerminalNode* SysYParser::InitValContext::COMMA(size_t i) {
  return getToken(SysYParser::COMMA, i);
}


size_t SysYParser::InitValContext::getRuleIndex() const {
  return SysYParser::RuleInitVal;
}


std::any SysYParser::InitValContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitInitVal(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::InitValContext* SysYParser::initVal() {
  InitValContext *_localctx = _tracker.createInstance<InitValContext>(_ctx, getState());
  enterRule(_localctx, 16, SysYParser::RuleInitVal);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(155);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SysYParser::L_PARENTHESIS:
      case SysYParser::UNARYOP:
      case SysYParser::IDENTIFIER:
      case SysYParser::INTEGER_CONST:
      case SysYParser::FLOATING_CONST: {
        enterOuterAlt(_localctx, 1);
        setState(142);
        exp();
        break;
      }

      case SysYParser::L_BRACE: {
        enterOuterAlt(_localctx, 2);
        setState(143);
        match(SysYParser::L_BRACE);
        setState(152);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & 940573760) != 0)) {
          setState(144);
          initVal();
          setState(149);
          _errHandler->sync(this);
          _la = _input->LA(1);
          while (_la == SysYParser::COMMA) {
            setState(145);
            match(SysYParser::COMMA);
            setState(146);
            initVal();
            setState(151);
            _errHandler->sync(this);
            _la = _input->LA(1);
          }
        }
        setState(154);
        match(SysYParser::R_BRACE);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncDefContext ------------------------------------------------------------------

SysYParser::FuncDefContext::FuncDefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::FuncTypeContext* SysYParser::FuncDefContext::funcType() {
  return getRuleContext<SysYParser::FuncTypeContext>(0);
}

tree::TerminalNode* SysYParser::FuncDefContext::IDENTIFIER() {
  return getToken(SysYParser::IDENTIFIER, 0);
}

tree::TerminalNode* SysYParser::FuncDefContext::L_PARENTHESIS() {
  return getToken(SysYParser::L_PARENTHESIS, 0);
}

tree::TerminalNode* SysYParser::FuncDefContext::R_PARENTHESIS() {
  return getToken(SysYParser::R_PARENTHESIS, 0);
}

SysYParser::BlockContext* SysYParser::FuncDefContext::block() {
  return getRuleContext<SysYParser::BlockContext>(0);
}

SysYParser::FuncFParamsContext* SysYParser::FuncDefContext::funcFParams() {
  return getRuleContext<SysYParser::FuncFParamsContext>(0);
}


size_t SysYParser::FuncDefContext::getRuleIndex() const {
  return SysYParser::RuleFuncDef;
}


std::any SysYParser::FuncDefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitFuncDef(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::FuncDefContext* SysYParser::funcDef() {
  FuncDefContext *_localctx = _tracker.createInstance<FuncDefContext>(_ctx, getState());
  enterRule(_localctx, 18, SysYParser::RuleFuncDef);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(157);
    funcType();
    setState(158);
    match(SysYParser::IDENTIFIER);
    setState(159);
    match(SysYParser::L_PARENTHESIS);
    setState(161);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SysYParser::INTTYPE

    || _la == SysYParser::FLOATTYPE) {
      setState(160);
      funcFParams();
    }
    setState(163);
    match(SysYParser::R_PARENTHESIS);
    setState(164);
    block();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncTypeContext ------------------------------------------------------------------

SysYParser::FuncTypeContext::FuncTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::FuncTypeContext::INTTYPE() {
  return getToken(SysYParser::INTTYPE, 0);
}

tree::TerminalNode* SysYParser::FuncTypeContext::FLOATTYPE() {
  return getToken(SysYParser::FLOATTYPE, 0);
}

tree::TerminalNode* SysYParser::FuncTypeContext::VOIDTYPE() {
  return getToken(SysYParser::VOIDTYPE, 0);
}


size_t SysYParser::FuncTypeContext::getRuleIndex() const {
  return SysYParser::RuleFuncType;
}


std::any SysYParser::FuncTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitFuncType(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::FuncTypeContext* SysYParser::funcType() {
  FuncTypeContext *_localctx = _tracker.createInstance<FuncTypeContext>(_ctx, getState());
  enterRule(_localctx, 20, SysYParser::RuleFuncType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(166);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 8216) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncFParamsContext ------------------------------------------------------------------

SysYParser::FuncFParamsContext::FuncFParamsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<SysYParser::FuncFParamContext *> SysYParser::FuncFParamsContext::funcFParam() {
  return getRuleContexts<SysYParser::FuncFParamContext>();
}

SysYParser::FuncFParamContext* SysYParser::FuncFParamsContext::funcFParam(size_t i) {
  return getRuleContext<SysYParser::FuncFParamContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::FuncFParamsContext::COMMA() {
  return getTokens(SysYParser::COMMA);
}

tree::TerminalNode* SysYParser::FuncFParamsContext::COMMA(size_t i) {
  return getToken(SysYParser::COMMA, i);
}


size_t SysYParser::FuncFParamsContext::getRuleIndex() const {
  return SysYParser::RuleFuncFParams;
}


std::any SysYParser::FuncFParamsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitFuncFParams(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::FuncFParamsContext* SysYParser::funcFParams() {
  FuncFParamsContext *_localctx = _tracker.createInstance<FuncFParamsContext>(_ctx, getState());
  enterRule(_localctx, 22, SysYParser::RuleFuncFParams);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(168);
    funcFParam();
    setState(173);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SysYParser::COMMA) {
      setState(169);
      match(SysYParser::COMMA);
      setState(170);
      funcFParam();
      setState(175);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncFParamContext ------------------------------------------------------------------

SysYParser::FuncFParamContext::FuncFParamContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::BTypeContext* SysYParser::FuncFParamContext::bType() {
  return getRuleContext<SysYParser::BTypeContext>(0);
}

tree::TerminalNode* SysYParser::FuncFParamContext::IDENTIFIER() {
  return getToken(SysYParser::IDENTIFIER, 0);
}

std::vector<tree::TerminalNode *> SysYParser::FuncFParamContext::L_BRACKET() {
  return getTokens(SysYParser::L_BRACKET);
}

tree::TerminalNode* SysYParser::FuncFParamContext::L_BRACKET(size_t i) {
  return getToken(SysYParser::L_BRACKET, i);
}

std::vector<tree::TerminalNode *> SysYParser::FuncFParamContext::R_BRACKET() {
  return getTokens(SysYParser::R_BRACKET);
}

tree::TerminalNode* SysYParser::FuncFParamContext::R_BRACKET(size_t i) {
  return getToken(SysYParser::R_BRACKET, i);
}

std::vector<SysYParser::ExpContext *> SysYParser::FuncFParamContext::exp() {
  return getRuleContexts<SysYParser::ExpContext>();
}

SysYParser::ExpContext* SysYParser::FuncFParamContext::exp(size_t i) {
  return getRuleContext<SysYParser::ExpContext>(i);
}


size_t SysYParser::FuncFParamContext::getRuleIndex() const {
  return SysYParser::RuleFuncFParam;
}


std::any SysYParser::FuncFParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitFuncFParam(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::FuncFParamContext* SysYParser::funcFParam() {
  FuncFParamContext *_localctx = _tracker.createInstance<FuncFParamContext>(_ctx, getState());
  enterRule(_localctx, 24, SysYParser::RuleFuncFParam);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(176);
    bType();
    setState(177);
    match(SysYParser::IDENTIFIER);
    setState(189);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == SysYParser::L_BRACKET) {
      setState(178);
      match(SysYParser::L_BRACKET);
      setState(179);
      match(SysYParser::R_BRACKET);
      setState(186);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == SysYParser::L_BRACKET) {
        setState(180);
        match(SysYParser::L_BRACKET);
        setState(181);
        exp();
        setState(182);
        match(SysYParser::R_BRACKET);
        setState(188);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockContext ------------------------------------------------------------------

SysYParser::BlockContext::BlockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::BlockContext::L_BRACE() {
  return getToken(SysYParser::L_BRACE, 0);
}

tree::TerminalNode* SysYParser::BlockContext::R_BRACE() {
  return getToken(SysYParser::R_BRACE, 0);
}

std::vector<SysYParser::BlockItemContext *> SysYParser::BlockContext::blockItem() {
  return getRuleContexts<SysYParser::BlockItemContext>();
}

SysYParser::BlockItemContext* SysYParser::BlockContext::blockItem(size_t i) {
  return getRuleContext<SysYParser::BlockItemContext>(i);
}


size_t SysYParser::BlockContext::getRuleIndex() const {
  return SysYParser::RuleBlock;
}


std::any SysYParser::BlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitBlock(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::BlockContext* SysYParser::block() {
  BlockContext *_localctx = _tracker.createInstance<BlockContext>(_ctx, getState());
  enterRule(_localctx, 26, SysYParser::RuleBlock);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(191);
    match(SysYParser::L_BRACE);
    setState(195);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 941573244) != 0)) {
      setState(192);
      blockItem();
      setState(197);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(198);
    match(SysYParser::R_BRACE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockItemContext ------------------------------------------------------------------

SysYParser::BlockItemContext::BlockItemContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::DeclContext* SysYParser::BlockItemContext::decl() {
  return getRuleContext<SysYParser::DeclContext>(0);
}

SysYParser::StmtContext* SysYParser::BlockItemContext::stmt() {
  return getRuleContext<SysYParser::StmtContext>(0);
}


size_t SysYParser::BlockItemContext::getRuleIndex() const {
  return SysYParser::RuleBlockItem;
}


std::any SysYParser::BlockItemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitBlockItem(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::BlockItemContext* SysYParser::blockItem() {
  BlockItemContext *_localctx = _tracker.createInstance<BlockItemContext>(_ctx, getState());
  enterRule(_localctx, 28, SysYParser::RuleBlockItem);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(202);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SysYParser::CONSTLABEL:
      case SysYParser::INTTYPE:
      case SysYParser::FLOATTYPE: {
        enterOuterAlt(_localctx, 1);
        setState(200);
        decl();
        break;
      }

      case SysYParser::SEMICOLON:
      case SysYParser::L_PARENTHESIS:
      case SysYParser::L_BRACE:
      case SysYParser::IFKEY:
      case SysYParser::WHILEKEY:
      case SysYParser::BREAKKEY:
      case SysYParser::CONTINUEKEY:
      case SysYParser::RETURNKEY:
      case SysYParser::UNARYOP:
      case SysYParser::IDENTIFIER:
      case SysYParser::INTEGER_CONST:
      case SysYParser::FLOATING_CONST: {
        enterOuterAlt(_localctx, 2);
        setState(201);
        stmt();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StmtContext ------------------------------------------------------------------

SysYParser::StmtContext::StmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::LValContext* SysYParser::StmtContext::lVal() {
  return getRuleContext<SysYParser::LValContext>(0);
}

tree::TerminalNode* SysYParser::StmtContext::ASSIGNMARK() {
  return getToken(SysYParser::ASSIGNMARK, 0);
}

SysYParser::ExpContext* SysYParser::StmtContext::exp() {
  return getRuleContext<SysYParser::ExpContext>(0);
}

tree::TerminalNode* SysYParser::StmtContext::SEMICOLON() {
  return getToken(SysYParser::SEMICOLON, 0);
}

SysYParser::BlockContext* SysYParser::StmtContext::block() {
  return getRuleContext<SysYParser::BlockContext>(0);
}

tree::TerminalNode* SysYParser::StmtContext::IFKEY() {
  return getToken(SysYParser::IFKEY, 0);
}

tree::TerminalNode* SysYParser::StmtContext::L_PARENTHESIS() {
  return getToken(SysYParser::L_PARENTHESIS, 0);
}

SysYParser::CondContext* SysYParser::StmtContext::cond() {
  return getRuleContext<SysYParser::CondContext>(0);
}

tree::TerminalNode* SysYParser::StmtContext::R_PARENTHESIS() {
  return getToken(SysYParser::R_PARENTHESIS, 0);
}

std::vector<SysYParser::StmtContext *> SysYParser::StmtContext::stmt() {
  return getRuleContexts<SysYParser::StmtContext>();
}

SysYParser::StmtContext* SysYParser::StmtContext::stmt(size_t i) {
  return getRuleContext<SysYParser::StmtContext>(i);
}

tree::TerminalNode* SysYParser::StmtContext::ELSEKEY() {
  return getToken(SysYParser::ELSEKEY, 0);
}

tree::TerminalNode* SysYParser::StmtContext::WHILEKEY() {
  return getToken(SysYParser::WHILEKEY, 0);
}

tree::TerminalNode* SysYParser::StmtContext::BREAKKEY() {
  return getToken(SysYParser::BREAKKEY, 0);
}

tree::TerminalNode* SysYParser::StmtContext::CONTINUEKEY() {
  return getToken(SysYParser::CONTINUEKEY, 0);
}

tree::TerminalNode* SysYParser::StmtContext::RETURNKEY() {
  return getToken(SysYParser::RETURNKEY, 0);
}


size_t SysYParser::StmtContext::getRuleIndex() const {
  return SysYParser::RuleStmt;
}


std::any SysYParser::StmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitStmt(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::StmtContext* SysYParser::stmt() {
  StmtContext *_localctx = _tracker.createInstance<StmtContext>(_ctx, getState());
  enterRule(_localctx, 30, SysYParser::RuleStmt);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(238);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(204);
      lVal();
      setState(205);
      match(SysYParser::ASSIGNMARK);
      setState(206);
      exp();
      setState(207);
      match(SysYParser::SEMICOLON);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(210);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 940572736) != 0)) {
        setState(209);
        exp();
      }
      setState(212);
      match(SysYParser::SEMICOLON);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(213);
      block();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(214);
      match(SysYParser::IFKEY);
      setState(215);
      match(SysYParser::L_PARENTHESIS);
      setState(216);
      cond();
      setState(217);
      match(SysYParser::R_PARENTHESIS);
      setState(218);
      stmt();
      setState(221);
      _errHandler->sync(this);

      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx)) {
      case 1: {
        setState(219);
        match(SysYParser::ELSEKEY);
        setState(220);
        stmt();
        break;
      }

      default:
        break;
      }
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(223);
      match(SysYParser::WHILEKEY);
      setState(224);
      match(SysYParser::L_PARENTHESIS);
      setState(225);
      cond();
      setState(226);
      match(SysYParser::R_PARENTHESIS);
      setState(227);
      stmt();
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(229);
      match(SysYParser::BREAKKEY);
      setState(230);
      match(SysYParser::SEMICOLON);
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(231);
      match(SysYParser::CONTINUEKEY);
      setState(232);
      match(SysYParser::SEMICOLON);
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(233);
      match(SysYParser::RETURNKEY);
      setState(235);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 940572736) != 0)) {
        setState(234);
        exp();
      }
      setState(237);
      match(SysYParser::SEMICOLON);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpContext ------------------------------------------------------------------

SysYParser::ExpContext::ExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::AddExpContext* SysYParser::ExpContext::addExp() {
  return getRuleContext<SysYParser::AddExpContext>(0);
}


size_t SysYParser::ExpContext::getRuleIndex() const {
  return SysYParser::RuleExp;
}


std::any SysYParser::ExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitExp(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::ExpContext* SysYParser::exp() {
  ExpContext *_localctx = _tracker.createInstance<ExpContext>(_ctx, getState());
  enterRule(_localctx, 32, SysYParser::RuleExp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(240);
    addExp(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CondContext ------------------------------------------------------------------

SysYParser::CondContext::CondContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::LOrExpContext* SysYParser::CondContext::lOrExp() {
  return getRuleContext<SysYParser::LOrExpContext>(0);
}


size_t SysYParser::CondContext::getRuleIndex() const {
  return SysYParser::RuleCond;
}


std::any SysYParser::CondContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitCond(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::CondContext* SysYParser::cond() {
  CondContext *_localctx = _tracker.createInstance<CondContext>(_ctx, getState());
  enterRule(_localctx, 34, SysYParser::RuleCond);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(242);
    lOrExp(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LValContext ------------------------------------------------------------------

SysYParser::LValContext::LValContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::LValContext::IDENTIFIER() {
  return getToken(SysYParser::IDENTIFIER, 0);
}

std::vector<tree::TerminalNode *> SysYParser::LValContext::L_BRACKET() {
  return getTokens(SysYParser::L_BRACKET);
}

tree::TerminalNode* SysYParser::LValContext::L_BRACKET(size_t i) {
  return getToken(SysYParser::L_BRACKET, i);
}

std::vector<SysYParser::ExpContext *> SysYParser::LValContext::exp() {
  return getRuleContexts<SysYParser::ExpContext>();
}

SysYParser::ExpContext* SysYParser::LValContext::exp(size_t i) {
  return getRuleContext<SysYParser::ExpContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::LValContext::R_BRACKET() {
  return getTokens(SysYParser::R_BRACKET);
}

tree::TerminalNode* SysYParser::LValContext::R_BRACKET(size_t i) {
  return getToken(SysYParser::R_BRACKET, i);
}


size_t SysYParser::LValContext::getRuleIndex() const {
  return SysYParser::RuleLVal;
}


std::any SysYParser::LValContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitLVal(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::LValContext* SysYParser::lVal() {
  LValContext *_localctx = _tracker.createInstance<LValContext>(_ctx, getState());
  enterRule(_localctx, 36, SysYParser::RuleLVal);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(244);
    match(SysYParser::IDENTIFIER);
    setState(251);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(245);
        match(SysYParser::L_BRACKET);
        setState(246);
        exp();
        setState(247);
        match(SysYParser::R_BRACKET); 
      }
      setState(253);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrimaryExpContext ------------------------------------------------------------------

SysYParser::PrimaryExpContext::PrimaryExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::PrimaryExpContext::L_PARENTHESIS() {
  return getToken(SysYParser::L_PARENTHESIS, 0);
}

SysYParser::ExpContext* SysYParser::PrimaryExpContext::exp() {
  return getRuleContext<SysYParser::ExpContext>(0);
}

tree::TerminalNode* SysYParser::PrimaryExpContext::R_PARENTHESIS() {
  return getToken(SysYParser::R_PARENTHESIS, 0);
}

SysYParser::LValContext* SysYParser::PrimaryExpContext::lVal() {
  return getRuleContext<SysYParser::LValContext>(0);
}

SysYParser::NumberContext* SysYParser::PrimaryExpContext::number() {
  return getRuleContext<SysYParser::NumberContext>(0);
}


size_t SysYParser::PrimaryExpContext::getRuleIndex() const {
  return SysYParser::RulePrimaryExp;
}


std::any SysYParser::PrimaryExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitPrimaryExp(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::PrimaryExpContext* SysYParser::primaryExp() {
  PrimaryExpContext *_localctx = _tracker.createInstance<PrimaryExpContext>(_ctx, getState());
  enterRule(_localctx, 38, SysYParser::RulePrimaryExp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(260);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case SysYParser::L_PARENTHESIS: {
        enterOuterAlt(_localctx, 1);
        setState(254);
        match(SysYParser::L_PARENTHESIS);
        setState(255);
        exp();
        setState(256);
        match(SysYParser::R_PARENTHESIS);
        break;
      }

      case SysYParser::IDENTIFIER: {
        enterOuterAlt(_localctx, 2);
        setState(258);
        lVal();
        break;
      }

      case SysYParser::INTEGER_CONST:
      case SysYParser::FLOATING_CONST: {
        enterOuterAlt(_localctx, 3);
        setState(259);
        number();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NumberContext ------------------------------------------------------------------

SysYParser::NumberContext::NumberContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::NumberContext::INTEGER_CONST() {
  return getToken(SysYParser::INTEGER_CONST, 0);
}

tree::TerminalNode* SysYParser::NumberContext::FLOATING_CONST() {
  return getToken(SysYParser::FLOATING_CONST, 0);
}


size_t SysYParser::NumberContext::getRuleIndex() const {
  return SysYParser::RuleNumber;
}


std::any SysYParser::NumberContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitNumber(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::NumberContext* SysYParser::number() {
  NumberContext *_localctx = _tracker.createInstance<NumberContext>(_ctx, getState());
  enterRule(_localctx, 40, SysYParser::RuleNumber);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(262);
    _la = _input->LA(1);
    if (!(_la == SysYParser::INTEGER_CONST

    || _la == SysYParser::FLOATING_CONST)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryExpContext ------------------------------------------------------------------

SysYParser::UnaryExpContext::UnaryExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::PrimaryExpContext* SysYParser::UnaryExpContext::primaryExp() {
  return getRuleContext<SysYParser::PrimaryExpContext>(0);
}

tree::TerminalNode* SysYParser::UnaryExpContext::IDENTIFIER() {
  return getToken(SysYParser::IDENTIFIER, 0);
}

tree::TerminalNode* SysYParser::UnaryExpContext::L_PARENTHESIS() {
  return getToken(SysYParser::L_PARENTHESIS, 0);
}

tree::TerminalNode* SysYParser::UnaryExpContext::R_PARENTHESIS() {
  return getToken(SysYParser::R_PARENTHESIS, 0);
}

SysYParser::FuncFParamsContext* SysYParser::UnaryExpContext::funcFParams() {
  return getRuleContext<SysYParser::FuncFParamsContext>(0);
}

SysYParser::UnaryOpContext* SysYParser::UnaryExpContext::unaryOp() {
  return getRuleContext<SysYParser::UnaryOpContext>(0);
}

SysYParser::UnaryExpContext* SysYParser::UnaryExpContext::unaryExp() {
  return getRuleContext<SysYParser::UnaryExpContext>(0);
}


size_t SysYParser::UnaryExpContext::getRuleIndex() const {
  return SysYParser::RuleUnaryExp;
}


std::any SysYParser::UnaryExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitUnaryExp(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::UnaryExpContext* SysYParser::unaryExp() {
  UnaryExpContext *_localctx = _tracker.createInstance<UnaryExpContext>(_ctx, getState());
  enterRule(_localctx, 42, SysYParser::RuleUnaryExp);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(274);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 27, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(264);
      primaryExp();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(265);
      match(SysYParser::IDENTIFIER);
      setState(266);
      match(SysYParser::L_PARENTHESIS);
      setState(268);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == SysYParser::INTTYPE

      || _la == SysYParser::FLOATTYPE) {
        setState(267);
        funcFParams();
      }
      setState(270);
      match(SysYParser::R_PARENTHESIS);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(271);
      unaryOp();
      setState(272);
      unaryExp();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryOpContext ------------------------------------------------------------------

SysYParser::UnaryOpContext::UnaryOpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* SysYParser::UnaryOpContext::UNARYOP() {
  return getToken(SysYParser::UNARYOP, 0);
}


size_t SysYParser::UnaryOpContext::getRuleIndex() const {
  return SysYParser::RuleUnaryOp;
}


std::any SysYParser::UnaryOpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitUnaryOp(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::UnaryOpContext* SysYParser::unaryOp() {
  UnaryOpContext *_localctx = _tracker.createInstance<UnaryOpContext>(_ctx, getState());
  enterRule(_localctx, 44, SysYParser::RuleUnaryOp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(276);
    match(SysYParser::UNARYOP);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncRParamsContext ------------------------------------------------------------------

SysYParser::FuncRParamsContext::FuncRParamsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<SysYParser::ExpContext *> SysYParser::FuncRParamsContext::exp() {
  return getRuleContexts<SysYParser::ExpContext>();
}

SysYParser::ExpContext* SysYParser::FuncRParamsContext::exp(size_t i) {
  return getRuleContext<SysYParser::ExpContext>(i);
}

std::vector<tree::TerminalNode *> SysYParser::FuncRParamsContext::COMMA() {
  return getTokens(SysYParser::COMMA);
}

tree::TerminalNode* SysYParser::FuncRParamsContext::COMMA(size_t i) {
  return getToken(SysYParser::COMMA, i);
}


size_t SysYParser::FuncRParamsContext::getRuleIndex() const {
  return SysYParser::RuleFuncRParams;
}


std::any SysYParser::FuncRParamsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitFuncRParams(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::FuncRParamsContext* SysYParser::funcRParams() {
  FuncRParamsContext *_localctx = _tracker.createInstance<FuncRParamsContext>(_ctx, getState());
  enterRule(_localctx, 46, SysYParser::RuleFuncRParams);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(278);
    exp();
    setState(283);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == SysYParser::COMMA) {
      setState(279);
      match(SysYParser::COMMA);
      setState(280);
      exp();
      setState(285);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MulExpContext ------------------------------------------------------------------

SysYParser::MulExpContext::MulExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::UnaryExpContext* SysYParser::MulExpContext::unaryExp() {
  return getRuleContext<SysYParser::UnaryExpContext>(0);
}

SysYParser::MulExpContext* SysYParser::MulExpContext::mulExp() {
  return getRuleContext<SysYParser::MulExpContext>(0);
}

tree::TerminalNode* SysYParser::MulExpContext::MULOP() {
  return getToken(SysYParser::MULOP, 0);
}


size_t SysYParser::MulExpContext::getRuleIndex() const {
  return SysYParser::RuleMulExp;
}


std::any SysYParser::MulExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitMulExp(this);
  else
    return visitor->visitChildren(this);
}


SysYParser::MulExpContext* SysYParser::mulExp() {
   return mulExp(0);
}

SysYParser::MulExpContext* SysYParser::mulExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::MulExpContext *_localctx = _tracker.createInstance<MulExpContext>(_ctx, parentState);
  SysYParser::MulExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 48;
  enterRecursionRule(_localctx, 48, SysYParser::RuleMulExp, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(287);
    unaryExp();
    _ctx->stop = _input->LT(-1);
    setState(294);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 29, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<MulExpContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleMulExp);
        setState(289);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(290);
        match(SysYParser::MULOP);
        setState(291);
        unaryExp(); 
      }
      setState(296);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 29, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- AddExpContext ------------------------------------------------------------------

SysYParser::AddExpContext::AddExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::MulExpContext* SysYParser::AddExpContext::mulExp() {
  return getRuleContext<SysYParser::MulExpContext>(0);
}

SysYParser::AddExpContext* SysYParser::AddExpContext::addExp() {
  return getRuleContext<SysYParser::AddExpContext>(0);
}

tree::TerminalNode* SysYParser::AddExpContext::ADDOP() {
  return getToken(SysYParser::ADDOP, 0);
}


size_t SysYParser::AddExpContext::getRuleIndex() const {
  return SysYParser::RuleAddExp;
}


std::any SysYParser::AddExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitAddExp(this);
  else
    return visitor->visitChildren(this);
}


SysYParser::AddExpContext* SysYParser::addExp() {
   return addExp(0);
}

SysYParser::AddExpContext* SysYParser::addExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::AddExpContext *_localctx = _tracker.createInstance<AddExpContext>(_ctx, parentState);
  SysYParser::AddExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 50;
  enterRecursionRule(_localctx, 50, SysYParser::RuleAddExp, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(298);
    mulExp(0);
    _ctx->stop = _input->LT(-1);
    setState(305);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<AddExpContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleAddExp);
        setState(300);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(301);
        match(SysYParser::ADDOP);
        setState(302);
        mulExp(0); 
      }
      setState(307);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- RelExpContext ------------------------------------------------------------------

SysYParser::RelExpContext::RelExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::AddExpContext* SysYParser::RelExpContext::addExp() {
  return getRuleContext<SysYParser::AddExpContext>(0);
}

SysYParser::RelExpContext* SysYParser::RelExpContext::relExp() {
  return getRuleContext<SysYParser::RelExpContext>(0);
}

tree::TerminalNode* SysYParser::RelExpContext::RELOP() {
  return getToken(SysYParser::RELOP, 0);
}


size_t SysYParser::RelExpContext::getRuleIndex() const {
  return SysYParser::RuleRelExp;
}


std::any SysYParser::RelExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitRelExp(this);
  else
    return visitor->visitChildren(this);
}


SysYParser::RelExpContext* SysYParser::relExp() {
   return relExp(0);
}

SysYParser::RelExpContext* SysYParser::relExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::RelExpContext *_localctx = _tracker.createInstance<RelExpContext>(_ctx, parentState);
  SysYParser::RelExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 52;
  enterRecursionRule(_localctx, 52, SysYParser::RuleRelExp, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(309);
    addExp(0);
    _ctx->stop = _input->LT(-1);
    setState(316);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 31, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<RelExpContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleRelExp);
        setState(311);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(312);
        match(SysYParser::RELOP);
        setState(313);
        addExp(0); 
      }
      setState(318);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 31, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- EqExpContext ------------------------------------------------------------------

SysYParser::EqExpContext::EqExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::RelExpContext* SysYParser::EqExpContext::relExp() {
  return getRuleContext<SysYParser::RelExpContext>(0);
}

SysYParser::EqExpContext* SysYParser::EqExpContext::eqExp() {
  return getRuleContext<SysYParser::EqExpContext>(0);
}

tree::TerminalNode* SysYParser::EqExpContext::EQOP() {
  return getToken(SysYParser::EQOP, 0);
}


size_t SysYParser::EqExpContext::getRuleIndex() const {
  return SysYParser::RuleEqExp;
}


std::any SysYParser::EqExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitEqExp(this);
  else
    return visitor->visitChildren(this);
}


SysYParser::EqExpContext* SysYParser::eqExp() {
   return eqExp(0);
}

SysYParser::EqExpContext* SysYParser::eqExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::EqExpContext *_localctx = _tracker.createInstance<EqExpContext>(_ctx, parentState);
  SysYParser::EqExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 54;
  enterRecursionRule(_localctx, 54, SysYParser::RuleEqExp, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(320);
    relExp(0);
    _ctx->stop = _input->LT(-1);
    setState(327);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 32, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<EqExpContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleEqExp);
        setState(322);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(323);
        match(SysYParser::EQOP);
        setState(324);
        relExp(0); 
      }
      setState(329);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 32, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- LAndExpContext ------------------------------------------------------------------

SysYParser::LAndExpContext::LAndExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::EqExpContext* SysYParser::LAndExpContext::eqExp() {
  return getRuleContext<SysYParser::EqExpContext>(0);
}

SysYParser::LAndExpContext* SysYParser::LAndExpContext::lAndExp() {
  return getRuleContext<SysYParser::LAndExpContext>(0);
}

tree::TerminalNode* SysYParser::LAndExpContext::LANDOP() {
  return getToken(SysYParser::LANDOP, 0);
}


size_t SysYParser::LAndExpContext::getRuleIndex() const {
  return SysYParser::RuleLAndExp;
}


std::any SysYParser::LAndExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitLAndExp(this);
  else
    return visitor->visitChildren(this);
}


SysYParser::LAndExpContext* SysYParser::lAndExp() {
   return lAndExp(0);
}

SysYParser::LAndExpContext* SysYParser::lAndExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::LAndExpContext *_localctx = _tracker.createInstance<LAndExpContext>(_ctx, parentState);
  SysYParser::LAndExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 56;
  enterRecursionRule(_localctx, 56, SysYParser::RuleLAndExp, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(331);
    eqExp(0);
    _ctx->stop = _input->LT(-1);
    setState(338);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 33, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<LAndExpContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleLAndExp);
        setState(333);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(334);
        match(SysYParser::LANDOP);
        setState(335);
        eqExp(0); 
      }
      setState(340);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 33, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- LOrExpContext ------------------------------------------------------------------

SysYParser::LOrExpContext::LOrExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::LAndExpContext* SysYParser::LOrExpContext::lAndExp() {
  return getRuleContext<SysYParser::LAndExpContext>(0);
}

SysYParser::LOrExpContext* SysYParser::LOrExpContext::lOrExp() {
  return getRuleContext<SysYParser::LOrExpContext>(0);
}

tree::TerminalNode* SysYParser::LOrExpContext::LOROP() {
  return getToken(SysYParser::LOROP, 0);
}


size_t SysYParser::LOrExpContext::getRuleIndex() const {
  return SysYParser::RuleLOrExp;
}


std::any SysYParser::LOrExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitLOrExp(this);
  else
    return visitor->visitChildren(this);
}


SysYParser::LOrExpContext* SysYParser::lOrExp() {
   return lOrExp(0);
}

SysYParser::LOrExpContext* SysYParser::lOrExp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  SysYParser::LOrExpContext *_localctx = _tracker.createInstance<LOrExpContext>(_ctx, parentState);
  SysYParser::LOrExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 58;
  enterRecursionRule(_localctx, 58, SysYParser::RuleLOrExp, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(342);
    lAndExp(0);
    _ctx->stop = _input->LT(-1);
    setState(349);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 34, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<LOrExpContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleLOrExp);
        setState(344);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(345);
        match(SysYParser::LOROP);
        setState(346);
        lAndExp(0); 
      }
      setState(351);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 34, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- ConstExpContext ------------------------------------------------------------------

SysYParser::ConstExpContext::ConstExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

SysYParser::AddExpContext* SysYParser::ConstExpContext::addExp() {
  return getRuleContext<SysYParser::AddExpContext>(0);
}


size_t SysYParser::ConstExpContext::getRuleIndex() const {
  return SysYParser::RuleConstExp;
}


std::any SysYParser::ConstExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<SysYVisitor*>(visitor))
    return parserVisitor->visitConstExp(this);
  else
    return visitor->visitChildren(this);
}

SysYParser::ConstExpContext* SysYParser::constExp() {
  ConstExpContext *_localctx = _tracker.createInstance<ConstExpContext>(_ctx, getState());
  enterRule(_localctx, 60, SysYParser::RuleConstExp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(352);
    addExp(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool SysYParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 24: return mulExpSempred(antlrcpp::downCast<MulExpContext *>(context), predicateIndex);
    case 25: return addExpSempred(antlrcpp::downCast<AddExpContext *>(context), predicateIndex);
    case 26: return relExpSempred(antlrcpp::downCast<RelExpContext *>(context), predicateIndex);
    case 27: return eqExpSempred(antlrcpp::downCast<EqExpContext *>(context), predicateIndex);
    case 28: return lAndExpSempred(antlrcpp::downCast<LAndExpContext *>(context), predicateIndex);
    case 29: return lOrExpSempred(antlrcpp::downCast<LOrExpContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool SysYParser::mulExpSempred(MulExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool SysYParser::addExpSempred(AddExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 1: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool SysYParser::relExpSempred(RelExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 2: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool SysYParser::eqExpSempred(EqExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 3: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool SysYParser::lAndExpSempred(LAndExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 4: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool SysYParser::lOrExpSempred(LOrExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 5: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

void SysYParser::initialize() {
  ::antlr4::internal::call_once(sysyParserOnceFlag, sysyParserInitialize);
}
