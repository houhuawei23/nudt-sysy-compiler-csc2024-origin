### antlr4 Class Relationshape

```c++
/* antlr4 Class Relationshape */
// Token Classes

  Token <- WritableToken <- CommonToken

// Stream Classes

             /- CharStream <- ANTLRInputStream <- ANTLRFileStream
            /             \-- UnbufferedCharStream
  IntStream 
            \- BufferedTokenStream <- CommonTokenStream
             \- UnbufferedTokenStream

// Lexer and Parser Classes

  TokenSource <- Lexer <- ExpressionLexer
                /
            /--/                
  Recognizer 
            \--- Parser <- ExpressionParser

// Parse Trees and Rule Contexts

                             /- ErrorNode
            /- TerminalNode /-- TerminalNodempl <- ErrorNodelmpl
  ParseTree
            \- RuleContext <- ParserRuleContext <- CompUnitContext
```


### antlr4 api

```c++
antlr4::Token {
    getText() -> size_t
    getType()
    getLine()
    getCharPositionInLine()
    getChannel()
    getTokenIndex()
    getStartIndex()
    getStopIndex()
    getTokenSource() -> TokenSource*
    getInputStream() -> CharStream*
    toString() -> string
}

antlr4::IntStream { // Pure Virtual
    virtual void consume() = 0;
    virtual size_t LA(ssize_t i) = 0;
    virtual ssize_t mark() = 0;
    virtual void release(ssize_t marker) = 0;
    virtual size_t index() = 0;
    virtual void seek(size_t index) = 0;
    virtual size_t size() = 0;
    virtual std::string getSourceName() const = 0;
}

antlr4::RuleContext : public tree::ParseTree {
    bool is(const tree::ParseTree &parseTree);
    virtual int depth();
    virtual bool isEmpty();
    virtual misc::Interval getSourceInterval() override;

    virtual std::string getText() override;

    virtual size_t getRuleIndex() const;
    // return the outer alternative number used to match the input
    virtual size_t getAltNumber() const; 
    virtual std::string toStringTree(Parser *recog, bool pretty = false) override;
    virtual std::string toStringTree(bool pretty = false) override;
    virtual std::string toString() override;
}

antlr4::ParserRuleContext : public RuleContext {
    Token *start;
    Token *stop;
    virtual void copyFrom(ParserRuleContext *ctx);
    tree::TerminalNode* getToken(size_t ttype, std::size_t i) const;

    std::vector<tree::TerminalNode*> getTokens(size_t ttype) const;

    template<typename T>
    T* getRuleContext(size_t i) const {...};

    template<typename T>
    std::vector<T*> getRuleContexts() const {...};

    virtual misc::Interval getSourceInterval() override;

    Token* getStart() const;
    Token* getStop() const;

    virtual std::string toInfoString(Parser *recognizer);
}



antlr4::tree::ParseTree {
    ParseTree *parent;
    vector<ParseTree *> children;
    string toStringTree(bool pretty = false)
    string toString();
    string getText();
    misc::Interval getSourceInterval();
}


```

### antlr4 api

```C++
Token
    getType()	            // The token's type
    getLine()	            // The number of the line containing the token
    getText()	            // The text associated with the token
    getCharPositionInLine()	// The line position of the token's first character
    getTokenIndex()	        // The index of the token in the input stream
    getChannel()	        // The channel that provides the token to the parser
    getStartIndex()	        // The token's starting character index
    getStopIndex()	        // The token's last character index
    getTokenSource()	    // A pointer to the TokenSource that created the token
    getInputStream()	    // A pointer to the CharStream that provided the token's characters

IntStream
    consume()                   // Accesses and consumes the current element
    LA(ssize_t i)               // Reads the element i positions away
    mark()                      // Returns a handle that identifies a position in the stream
    release(ssize_t marker)     // Releases the position handle
    index()                     // Returns the index of the upcoming element
    seek(ssize_t index)         // Set the input cursor to the given position
    size()                      // Returns the number of elements in the stream


Parser
    /* Token Functions */
    consume()                       // Consumes and returns the current token
    getCurrentToken()               // Returns the current token
    isExpectedToken(size_t symbol)  // Checks whether the current token has the given type
    getExpectedTokens()             // Provides tokens in the current context
    isMatchedEOF()                  // Identifies the current token is EOF
    createTerminalNode(Token* t)    // Adds a new terminal node to the tree
    createErrorNode(Token* t)       // Adds a new error node to the tree
    match(size_t ttype)             // Returns the token if it matches the given type
    matchWildcard()                 // Match the current token as a wildcard
    getInputStream()                // Returns the parser's IntStream
    setInputStream(IntStream* is)   // Sets the parser's IntStream
    getTokenStream()                // Returns the parser's TokenStream
    setTokenStream(TokenStream* ts) // Sets the parser's TokenStream
    getTokenFactory()               // Returns the parser's TokenFactory
    reset()                         // Resets the parser's state

    /* Parse Tree and Listener Functions */
    getBuildParseTree()                         // Checks if a parse tree will be constructed during parsing
    setBuildParseTree(bool b)                   // Identifies if a parse tree should be constructed
    getTrimParseTree()                          // Checks if the parse tree is trimmed during parsing
    setTrimParseTree(bool t)                    // Identifies the parse tree should be trimmed during parsing
    getParseListeners()                         // Returns the vector containing the parser's listeners
    addParseListener(ParseTreeListener* ptl)    // Adds a listener to the parser
    removeParseListener(ParseTreeListener* ptl) // Removes a listener from the parser
    removeParseListeners()                      // Removes all listeners from the parser

    /* Error Functions */
    getNumberOfSyntaxErrors()                   // Returns the number of syntax errors
    getErrorHandler()                           // Returns the parser's error handler
    setErrorHandler(handler)                    // Sets the parser's error handler
    notifyErrorListeners(string msg)            // Sends a message to the parser's error listeners
    notifyErrorListeners(Token* t, string msg, exception_ptr e)  // Sends data to the parser's error listeners

    /* Rule Functions */
    enterRule(ParserRuleContext* ctx, size_t state, size_t index)  // Called upon rule entry
    exitRule()                                  // Called upon rule exit
    triggerEnterRuleEvent()                     // Notify listeners of rule entry
    triggerExitRuleEvent()                      // Notify listeners of rule exit
    getRuleIndex(string rulename)               // Identify the index of the given rule
    getPrecedence()                             // Get the precedence level of the topmost rule
    getContext()                                // Returns the context of the current rule
    setContext(ParserRuleContext* ctx)          // Sets the parser's current rule
    getInvokingContext(size_t index)            // Returns the context that invoked the current context
    getRuleInvocationStack()                    // Returns a list of rules processed up to the current rule
    getRuleInvocationStack(RuleContext* ctx)    // Returns a list of rules processed up to the given rule
```