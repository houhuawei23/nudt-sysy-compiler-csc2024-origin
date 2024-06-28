#pragma once
#include <cassert>
#include <forward_list>
#include <unordered_map>
#include "ir/value.hpp"

namespace ir {
//! SymbolTableBeta

/**
 * @brief
 * XXScope new_scope(_tables)
 * .lookup(name)
 * .insert(name, value)
 */
class SymbolTableBeta {
   private:
    enum Kind {
        _Module,
        _Function,
        _Block,
    };

   public:
    /*
     * @brief 符号表作用域 (Module, Function, Block)
     */
    struct ModuleScope {
        SymbolTableBeta& tables_ref;
        ModuleScope(SymbolTableBeta& tables) : tables_ref(tables) {
            tables.enter(_Module);
        }
        ~ModuleScope() { tables_ref.exit(); }
    };
    struct FunctionScope {
        SymbolTableBeta& tables_ref;
        FunctionScope(SymbolTableBeta& tables) : tables_ref(tables) {
            tables.enter(_Function);
        }
        ~FunctionScope() { tables_ref.exit(); }
    };
    struct BlockScope {
        SymbolTableBeta& tables_ref;
        BlockScope(SymbolTableBeta& tables) : tables_ref(tables) {
            tables.enter(_Block);
        }
        ~BlockScope() { tables_ref.exit(); }
    };

   private:
    /*
     * @brief 按照作用域范围建立符号表 (作用域小的符号表 -> 作用域大的符号表,
     * forward_list组织, 每个作用域使用map建立符号表)
     */
    std::forward_list<std::pair<Kind, std::unordered_map<std::string, Value*>>>
        symbols;

   public:
    SymbolTableBeta() = default;

   private:
    /*
     * @brief 创造 or 销毁 某一作用域的符号表
     */
    void enter(Kind kind) {
        symbols.emplace_front();
        symbols.front().first = kind;
    }
    void exit() { symbols.pop_front(); }

   public:
    /*
     * @brief 判断属于哪部分的作用域
     */
    bool isModuleScope() const { return symbols.front().first == _Module; }
    bool isFunctionScope() const { return symbols.front().first == _Function; }
    bool isBlockScope() const { return symbols.front().first == _Block; }

    /*
     * @brief 查表 (从当前作用域开始查, 直至查到全局作用域范围)
     */
    Value* lookup(const_str_ref name) const {
        for (auto& scope : symbols) {
            auto iter = scope.second.find(name);
            if (iter != scope.second.end())
                return iter->second;
        }
        return nullptr;
    }

    /*
     * @brief 为当前作用域插入表项
     *   Return: pair<map<string, Value*>::iterator, bool>
     */
    auto insert(const_str_ref name, Value* value) {
        assert(not symbols.empty());
        return symbols.front().second.emplace(name, value);
    }
};
}  // namespace ir