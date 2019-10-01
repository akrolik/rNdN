#pragma once

#include "Runtime/DataObjects/ContiguousDataObject.h"

#include <string>
#include <vector>

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Runtime {

class DataVector : public ContiguousDataObject
{
public:
	static DataVector *CreateVector(const HorseIR::BasicType *type, unsigned long size);

	virtual size_t GetElementCount() const = 0;
	virtual size_t GetElementSize() const = 0;

	virtual std::string DebugDump(unsigned int index) const = 0;
};

template<typename T>
class TypedDataVector : public DataVector
{
public:
	TypedDataVector(const HorseIR::BasicType *elementType, const std::vector<T>& data) : m_type(elementType), m_data(data) {}

	TypedDataVector(const HorseIR::BasicType *elementType, unsigned long size) : m_type(elementType)
	{
		m_data.resize(size);
	}

	const HorseIR::BasicType *GetType() const { return m_type; }

	const T& GetValue(unsigned int i) const { return m_data.at(i); }
	const std::vector<T>& GetValues() const { return m_data; }


        void *GetData() override { return m_data.data(); }
	size_t GetDataSize() const override { return m_data.size() * sizeof(T); }

	size_t GetElementCount() const override { return m_data.size(); }
	size_t GetElementSize() const override { return sizeof(T); }

	std::string Description() const override
	{
		return (HorseIR::PrettyPrinter::PrettyString(m_type) + "(" + std::to_string(GetElementSize()) + " bytes) x " + std::to_string(GetElementCount()));
	}

	std::string DebugDump() const override
	{
		std::string string;
		auto count = std::min(m_data.size(), 10ul);
		if (count > 1)
		{
			string += "(";
		}
		bool first = true;
		for (auto i = 0u; i < count; i++)
		{
			if (!first)
			{
				string += ", ";
			}
			first = false;
			string += DebugDump(i);
		}
		if (m_data.size() > 10)
		{
			string += ", ...";
		}
		if (count > 1)
		{
			string += ")";
		}
		string += ":" + HorseIR::PrettyPrinter::PrettyString(m_type);
		return string;
	}

	std::string DebugDump(unsigned int index) const override
	{
		if constexpr(std::is_same<T, std::string>::value)
		{
			return m_data.at(index);
		}
		else if constexpr(std::is_pointer<T>::value)
		{
			std::stringstream stream;
			stream << *m_data.at(index);
			return stream.str();
		}
		else
		{
			return std::to_string(m_data.at(index));
		}
	}

private:
	const HorseIR::BasicType *m_type = nullptr;

	std::vector<T> m_data;
};

}
