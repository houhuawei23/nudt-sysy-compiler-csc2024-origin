#include "value.hpp"
#include <unordered_map>
#include <forward_list>
#include <cassert>

namespace ir
{
    //! SymbolTable
    class SymbolTable
    {
    private:
        enum Kind
        {
            _Module,
            _Function,
            _Block,
        };

    public:
        /*
         * @brief 符号表作用域 (Module, Function, Block)
         */
        struct ModuleScope
        {
            SymbolTable &table;
            ModuleScope(SymbolTable &table) : table(table) { table.enter(_Module); }
            ~ModuleScope() { table.exit(); }
        };
        struct FunctionScope
        {
            SymbolTable &table;
            FunctionScope(SymbolTable &table) : table(table) { table.enter(_Function); }
            ~FunctionScope() { table.exit(); }
        };
        struct BlockScope
        {
            SymbolTable &table;
            BlockScope(SymbolTable &table) : table(table) { table.enter(_Block); }
            ~BlockScope() { table.exit(); }
        };

    private:
        /*
         * @brief 按照作用域范围建立符号表 (作用域小的符号表 -> 作用域大的符号表, forward_list组织, 每个作用域使用map建立符号表)
         */
        std::forward_list<std::pair<Kind, std::unordered_map<std::string, Value *>>> symbols;

    public:
        SymbolTable() = default;

    private:
        /*
         * @brief 创造 or 销毁 某一作用域的符号表
         */
        void enter(Kind kind)
        {
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
        Value *lookup(const std::string &name) const
        {
            for (auto &scope : symbols)
            {
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
        auto insert(const std::string &name, Value *value)
        {
            assert(not symbols.empty());
            return symbols.front().second.emplace(name, value);
        }
    };
}