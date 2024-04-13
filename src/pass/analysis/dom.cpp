// idom create algorithm from paper by Lengauer and Tarjan
// paper name: A fast algorithm for finding dominators in a flowgraph
// by: Thomas Lengauer and Robert Endre Tarjan
#include "pass/analysis/dom.hpp"
#include<set>
#include<map>
#include<algorithm>

static std::unordered_map<ir::BasicBlock*,ir::BasicBlock*>parent;
static std::unordered_map<ir::BasicBlock*,int>semi;
static std::vector<ir::BasicBlock*>vertex;
using bbset=std::set<ir::BasicBlock*>;
static std::unordered_map<ir::BasicBlock*,bbset>bucket;
static std::unordered_map<ir::BasicBlock*,ir::BasicBlock*>idom;
static std::unordered_map<ir::BasicBlock*,ir::BasicBlock*>ancestor;
static std::unordered_map<ir::BasicBlock*,ir::BasicBlock*>child;
static std::unordered_map<ir::BasicBlock*,int>size;
static std::unordered_map<ir::BasicBlock*,ir::BasicBlock*>label;
static int dfc;




namespace pass
{
    //pre process for dom calc
    void preProcDom::run(ir::Function* func){
        if(!func->entry())return;
        auto blocklist=func->blocks();
        for(auto bbiter=blocklist.begin();bbiter!=blocklist.end();){
            auto bb=*bbiter;
            if(bb->pre_blocks().empty() and bb!=func->entry()){
                bbiter++;
                func->delete_block(bb);
            }
            else{
                bbiter++;
            }
        }
    }
    std::string preProcDom::name(){
        return "preProcDom";
    } 
    //LT algorithm to get idom and sdom
    void idomGen::compress(ir::BasicBlock* bb){
        auto ancestorBB=ancestor[bb];
        if(ancestor[ancestorBB]){
            compress(ancestorBB);
            if(semi[label[ancestorBB]]<semi[label[bb]]){
                label[bb]=label[ancestorBB];
            }
            ancestor[bb]=ancestor[ancestorBB];
        }
    }

    void idomGen::link(ir::BasicBlock* v,ir::BasicBlock* w){
        auto s=w;
        while(semi[label[w]]<semi[label[child[s]]]){
            if(size[s]+size[child[child[s]]]>=2*size[child[s]]){
                ancestor[child[s]]=s;
                child[s]=child[child[s]];
            }else{
                size[child[s]]=size[s];
                s=ancestor[s]=child[s];
            }
        }
        label[s]=label[w];
        size[v]=size[v]+size[w];
        if(size[v]<2*size[w]){
            auto tmp=s;
            s=child[v];
            child[v]=tmp;
        }
        while(s){
            ancestor[s]=v;
            s=child[s];
        }
    }

    ir::BasicBlock* idomGen::eval(ir::BasicBlock* bb){
        if(ancestor[bb]==0){
            return label[bb];
        }
        compress(bb);
        return (semi[label[ancestor[bb]]]>=semi[label[bb]])?label[bb]:label[ancestor[bb]];
    }

    void idomGen::dfsBlocks(ir::BasicBlock* bb){
        semi[bb]=dfc++;
        vertex.push_back(bb);
        for(auto bbnext:bb->next_blocks()){
            if(semi[bbnext]==0){
                parent[bbnext]=bb;
                dfsBlocks(bbnext);
            }
        }
    }

    void idomGen::run(ir::Function* func){
        if(!func->entry())return;
        parent.clear();
        semi.clear();
        vertex.clear();
        bucket.clear();
        idom.clear();
        ancestor.clear();
        child.clear();
        size.clear();
        label.clear();
        //step 1
        //initialize all arrays and maps
        for(auto bb:func->blocks()){
            semi[bb]=0;
            ancestor[bb]=nullptr;
            child[bb]=nullptr;
            label[bb]=bb;
            size[bb]=1;
        }
        semi[nullptr]=0;
        label[nullptr]=nullptr;
        size[nullptr]=0;
        //dfs
        dfc=0;// can't static def in dfs func, think about why
        dfsBlocks(func->entry());
        //step2 and 3
        for(auto bbIter=vertex.rbegin();bbIter!=vertex.rend();bbIter++){
            auto w=*bbIter;
            if(!parent[w])continue;
            
            for(auto v:w->pre_blocks()){
                auto u=eval(v);
                if(semi[u]<semi[w])semi[w]=semi[u];
            }
            bucket[vertex[semi[w]]].insert(w);
            link(parent[w],w);
            auto tmp=bucket[parent[w]];
            for(auto v:tmp){
                bucket[parent[w]].erase(v);
                auto u=eval(v);
                idom[v]=(semi[u]<semi[v])?u:parent[w];
            }
        }

        //step4
        for(auto bbIter=vertex.begin();bbIter!=vertex.end();bbIter++){
            auto w=*bbIter;
            if(idom[w]!=vertex[semi[w]])
                idom[w]=idom[idom[w]];
        }
        idom[func->entry()]=nullptr;

        //extra step, store informations into BasicBlocks
        for(auto bb:func->blocks()){
            bb->idom=idom[bb];
            bb->sdom=vertex[semi[bb]];
        }

    }

    std::string idomGen::name(){return "idomGen";}
    
    void domFrontierGen::getDomTree(ir::Function* func){
        for(auto bb : func->blocks())
            bb->domTree.clear();
        for(auto bb : func->blocks()){
            if(bb->idom)
                bb->idom->domTree.push_back(bb);
        }
    }

    void domFrontierGen::getDomInfo(ir::BasicBlock* bb, int level){
        bb->domLevel=level;
        for(auto bbnext:bb->domTree){
            getDomInfo(bbnext,level+1);
        }

    }

    void domFrontierGen::getDomFrontier(ir::Function* func){
        for(auto bb : func->blocks())
            bb->domFrontier.clear();
        for(auto bb : func->blocks()){
            if(bb->pre_blocks().size()>1){
                for(auto bbnext : bb->pre_blocks()){
                    auto runner=bbnext;
                    while(runner!=bb->idom){
                        runner->domFrontier.push_back(bb);
                        runner=runner->idom;
                    }
                }
            }
        }
    }

    //generate dom tree
    void domFrontierGen::run(ir::Function* func){
        if(!func->entry())return;
        getDomTree(func);
        getDomInfo(func->entry(),0);
        getDomFrontier(func);
        
    }


    std::string domFrontierGen::name(){return "domFrontierGen";}

    //debug info print pass
    void domInfoCheck::run(ir::Function* func){
        if(!func->entry())return;
        using namespace std;
        cout<<"In Function \""<<func->name()<<"\""<<endl;
        for(auto bb:func->blocks()){
            cout<<bb->name()<<" Prec: ";
            for(auto bbpre:bb->pre_blocks()){
                cout<<"\t"<<bbpre->name();
            }
            cout<<endl;
        }
        cout<<endl;
        for(auto bb:func->blocks()){
            cout<<bb->name()<<" Succ: ";
            for(auto bbnext:bb->next_blocks()){
                cout<<"\t"<<bbnext->name();
            }
            cout<<endl;
        }
        cout<<endl;
        for(auto bb:func->blocks()){
            cout<<bb->name()<<" idom: ";
            if(bb->idom)
                cout<<"\t"<<bb->idom->name();
            else
                cout<<"null";
            cout<<endl;
        }
        cout<<endl;
        for(auto bb:func->blocks()){
            cout<<bb->name()<<" sdom: ";
            if(bb->sdom)
                cout<<"\t"<<bb->sdom->name();
            else
                cout<<"null";
            cout<<endl;
        }
        cout<<endl;
        for(auto bb:func->blocks()){
            cout<<bb->name()<<" domTreeSons: ";
            for(auto bbson:bb->domTree){
                cout<<bbson->name()<<'\t';
            }
            cout<<endl;
        }
        cout<<endl;
        for(auto bb:func->blocks()){
            cout<<bb->name()<<" domFrontier: ";
            for(auto bbf:bb->domFrontier){
                cout<<bbf->name()<<'\t';
            }
            cout<<endl;
        }
    }
    std::string domInfoCheck::name(){
        return "domInfoCheck";
    }

} // namespace pass