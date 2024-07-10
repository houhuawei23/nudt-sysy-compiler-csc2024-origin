#pragma once
#include <string_view>

// __PRETTY_FUNCTION__ extension should be a constant expression
// Please refer to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=87372
// Workaround: only use constexpr with gcc>=9.4
#if (__GNUC__ > 9) || (__GNUC__ == 9 && __GNUC_MINOR__ >= 4)
#define CMMC_CONSTEXPR constexpr
#else
#define CMMC_CONSTEXPR
#endif

namespace utils {
template <typename Enum, Enum Value>
CMMC_CONSTEXPR const char* staticEnumNameImpl() {
    return __PRETTY_FUNCTION__;
}

template <typename Enum, Enum Value>
CMMC_CONSTEXPR std::string_view staticEnumName() {
    const std::string_view name = staticEnumNameImpl<Enum, Value>();
    const auto begin = name.find_last_of('=') + 2;
    return name.substr(begin, name.size() - begin - 1);
}

template <typename Enum, Enum Value>
CMMC_CONSTEXPR std::string_view enumName(Enum val) {
    if constexpr (static_cast<uint32_t>(Value) >= 128) { 
        // make clangd happy
        return "Unknown";
    } else {
        CMMC_CONSTEXPR auto name = staticEnumName<Enum, Value>();
        if CMMC_CONSTEXPR (name[0] == '(') {
            return "Unknown";
        }
        if (val == Value)
            return name;
        return enumName<Enum, static_cast<Enum>(static_cast<uint32_t>(Value) +
                                                1)>(val);
    }
}

template <typename Enum>
CMMC_CONSTEXPR std::string_view enumName(Enum val) {
    return utils::enumName<Enum, static_cast<Enum>(0)>(val);
}

template <typename T>
CMMC_CONSTEXPR const char* staticTypeNameImpl() {
    return __PRETTY_FUNCTION__;
}

template <typename T>
CMMC_CONSTEXPR std::string_view staticTypeName() {
    const std::string_view name = staticTypeNameImpl<T>();
    const auto begin = name.find_last_of('=') + 8;
    return name.substr(begin, name.size() - begin - 1);
}

template <typename T>
CMMC_CONSTEXPR std::string_view typeName() {
    return utils::staticTypeName<T>();
}

}  // namespace utils