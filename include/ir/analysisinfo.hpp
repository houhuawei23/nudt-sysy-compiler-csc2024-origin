#include "ir/infrast.hpp"
#include "ir/function.hpp"
#include <unordered_map>
#include <vector>

namespace ir{
    template<typename PassUnit> class analysisInfo;
    class domTree;
    class Loop;
    class loopInfo;
    class callGraph;
    
    template<typename PassUnit>
    class analysisInfo{
        private:
            PassUnit* _pu;
            bool _isvalid;
        public:
            analysisInfo(PassUnit*mp, bool v=false):_isvalid(v),_pu(mp){}
            void setOn() {_isvalid=true;}
            void setOff() {_isvalid=false;}
            virtual void refresh() = 0;
    };
    using ModuleACtx=analysisInfo<Module>;
    using FunctionACtx=analysisInfo<Function>;

    // add new analysis info of ir here!
    // dom Tree
    class domTree:public FunctionACtx{//also used as pdom
        private:
            std::unordered_map<BasicBlock*,BasicBlock*>_idom;
            std::unordered_map<BasicBlock*,std::vector<BasicBlock*>>_domson;
            std::unordered_map<BasicBlock*,std::vector<BasicBlock*>>_domfrontier;
        public:
            domTree(Function*func):FunctionACtx(func){}
            BasicBlock*idom(BasicBlock*bb){return _idom[bb];}
            std::vector<BasicBlock*>& domson(BasicBlock*bb){return _domson[bb];}
            std::vector<BasicBlock*>& domfrontier(BasicBlock*bb){return _domfrontier[bb];}
            void clearAll(){
                _idom.clear();
                _domson.clear();
                _domfrontier.clear();
            }
    };
    
   class Loop {
    private:
        std::set<BasicBlock*> _blocks;
        BasicBlock* _header;
        Function* _parent;
        std::set<BasicBlock*> _exits;

    public:
        Loop(BasicBlock* header, Function* parent) {
            _header = header;
            _parent = parent;
        }
        BasicBlock* header() { return _header; }
        Function* parent() { return _parent; }
        std::set<BasicBlock*>& blocks() { return _blocks; }
    };

    class loopInfo:public FunctionACtx{
        private:
            std::vector<Loop*>_loops;
            std::unordered_map<BasicBlock*,Loop*>_head2loop;
            std::unordered_map<BasicBlock*,int>looplevel;
        public:
            loopInfo(Function*fp):FunctionACtx(fp){}
            std::vector<Loop*>&loops(){return _loops;}
            Loop*head2loop(BasicBlock* bb){return _head2loop[bb];}
            void clearAll(){
                _loops.clear();
                _head2loop.clear();
            }
    };

    class callGraph:public ModuleACtx{
        private:
            std::unordered_map<Function*,std::vector<Function*>>_callees;
            std::unordered_map<Function*,bool>_is_called;
            std::unordered_map<Function*,bool>_is_inline;
            std::unordered_map<Function*,bool>_is_lib;
        public:
            callGraph(Module* md):ModuleACtx(md){}
            std::vector<Function*>&callees(Function*func){return _callees[func];}
            bool isCalled(Function*func){return _is_called[func];}
            bool isInline(Function*func){return _is_inline[func];}
            bool isLib(Function*func){return _is_lib[func];}
            void clearAll(){
                _callees.clear();
                _is_called.clear();
                _is_inline.clear();
                _is_lib.clear();
            }
    };
}