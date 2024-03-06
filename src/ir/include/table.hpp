
#include "value.hpp"
#include <unordered_map>

namespace ir {
/**
 * @brief symboltable used by building the module.
 *
 * each scope has its own symbol table, when enter the new scope,
 * will create a new symbol table and set parent point to the father scope.
 *
 * when lookup the table, it will search current table first,
 * if not find, will recursivly search throw parent link.
 *
 *
 */
class SymbolTable {
  private:
    SymbolTable *_father;
    std::unordered_map<std::string, ir::Value *> _table;

  public:
    SymbolTable() : _father(nullptr) {}
    SymbolTable(SymbolTable *father) : _father(father) {}

    SymbolTable *get_father() { return _father; }

    void set_father(SymbolTable *father) { _father = father; }

    void insert(std::string name, ir::Value *value) {
        _table.insert(std::make_pair(name, value));
    }

    // ir::Value *lookup(std::string name) {
    //     auto it = _table.find(name);
    //     if (it != _table.end()) {
    //         return it->second;
    //     }

    // }
    ir::Value *base_lookup(std::string name) {
        auto it = _table.find(name);
        if (it != _table.end()) {
            return it->second;
        }
        return nullptr;
    }
    ir::Value *lookup(std::string name) {
        SymbolTable *fptr = this;
        while (fptr != nullptr) {
            auto it = fptr->base_lookup(name);
            if (it != nullptr) {
                return it;
            }
            fptr = fptr->get_father();
        }
        return nullptr;
    }
};
} // namespace ir
