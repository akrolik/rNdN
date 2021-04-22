#pragma once

#include <utility>

#include "Utils/Logger.h"

#include "Libraries/robin_hood.h"

namespace Analysis {

template<typename T>
struct PointerValue
{
	struct Equals
	{
		 bool operator()(const T *val1, const T *val2) const
		 {
			 return val1 == val2;
		 }
	};
};

template<typename T>
struct Value
{
	struct Equals
	{
		 bool operator()(const T *val1, const T *val2) const
		 {
			 return *val1 == *val2;
		 }
	};
};

template<typename T>
struct Set : public robin_hood::unordered_set<const typename T::Type *, typename T::Hash, typename T::Equals>
{
	using robin_hood::unordered_set<const typename T::Type *, typename T::Hash, typename T::Equals>::unordered_set;

	void Print(std::ostream& os, unsigned int level = 0) const
	{
		os << std::string(level * Utils::Logger::IndentSize, ' ');
		auto first = true;
		for (const auto& val : *this)
		{
			if (!first)
			{
				os << ", ";
			}
			first = false;
			T::Print(os, val);
		}
	}

	bool operator==(const Set<T>& other) const
	{
		if (this->size() != other.size())
		{
			return false;
		}

		for (auto it = this->begin(); it != this->end(); ++it)
		{
			if (other.find(*it) == other.end())
			{
				return false;
			}
		}
		return true;
	}

	bool operator!=(const Set<T>& other) const
	{
		return !(*this == other);
	}
};

template<typename K, typename V>
struct Map : public robin_hood::unordered_map<const typename K::Type *, const typename V::Type *, typename K::Hash, typename K::Equals>
{
	void Print(std::ostream& os, unsigned int level = 0) const
	{
		auto first = true;
		for (const auto& pair : *this)
		{
			if (!first)
			{
				os << std::endl;
			}
			first = false;
			os << std::string(level * Utils::Logger::IndentSize, ' ');
			K::Print(os, pair.first);
			os << "->";
			V::Print(os, pair.second);
		}
	}

	bool operator==(const Map<K, V>& other) const
	{
		if (this->size() != other.size())
		{
			return false;
		}

		for (auto it = this->begin(); it != this->end(); ++it)
		{
			auto y = other.find(it->first);
			if (y == other.end() || !typename V::Equals()(y->second, it->second))
			{
				return false;
			}

		}
		return true;
	}

	bool operator!=(const Map<K, V>& other)const
	{
		return !(*this == other);
	}
};

template<typename T1, typename T2>
struct Pair : public std::pair<T1, T2>
{
	void clear() noexcept
	{
		this->first.clear();
		this->second.clear();
	}

	void Print(std::ostream& os, unsigned int level = 0) const
	{
		os << std::string(level * Utils::Logger::IndentSize, ' ');
		os << "<" << std::endl;
		this->first.Print(os, level + 1);
		os << std::endl;
		os << std::string(level * Utils::Logger::IndentSize, ' ');
		os << "," << std::endl;
		this->second.Print(os, level + 1);
		os << std::endl;
		os << std::string(level * Utils::Logger::IndentSize, ' ');
		os << ">";
	}
};

}
