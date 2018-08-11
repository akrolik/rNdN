#pragma once

#include "Runtime/DataObjects/DataObject.h"

#include <string>
#include <vector>

#include "HorseIR/Tree/Types/BasicType.h"

namespace Runtime {

class DataVector : public DataObject
{
public:
	static DataVector *CreateVector(HorseIR::BasicType *type, unsigned long size);

	virtual void *GetData() = 0;
	virtual size_t GetDataSize() const = 0;
	virtual size_t GetElementCount() const = 0;
	virtual size_t GetElementSize() const = 0;

	//TODO:
	void Dump() const override {}

	virtual std::string Dump(unsigned int index) const = 0;
};

template<typename T>
class TypedDataVector : public DataVector
{
public:
	TypedDataVector(HorseIR::BasicType *elementType, const std::vector<T>& data) : m_type(elementType), m_data(data) {}

	TypedDataVector(HorseIR::BasicType *elementType, unsigned long size) : m_type(elementType)
	{
		m_data.resize(size);
	}

	HorseIR::BasicType *GetType() const { return m_type; }

	const T& GetValue(unsigned int i) const { return m_data.at(i); }
	const std::vector<T>& GetValues() const { return m_data; }

	void *GetData() override { return m_data.data(); }
	size_t GetDataSize() const override { return m_data.size() * sizeof(T); }
	size_t GetElementCount() const override { return m_data.size(); }
	size_t GetElementSize() const override { return sizeof(T); }

	std::string Dump(unsigned int index) const override
	{
		if constexpr(std::is_same<T, std::string>::value)
		{
			return m_data.at(index);
		}
		else
		{
			return std::to_string(m_data.at(index));
		}
	}

private:
	HorseIR::BasicType *m_type = nullptr;

	std::vector<T> m_data;
};

}
