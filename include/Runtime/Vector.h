#pragma once

#include "Runtime/DataObject.h"

#include <string>
#include <vector>

#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"

namespace Runtime {

class Vector : public DataObject
{
public:
	static Vector *CreateVector(const HorseIR::PrimitiveType *type, unsigned long size);

	virtual const HorseIR::Type *GetType() const = 0;

	virtual void *GetData() = 0;
	virtual size_t GetCount() const = 0;
	virtual size_t GetSize() const = 0;
	virtual size_t GetElementSize() const = 0;

	//TODO:
	void Dump() const override {}

	virtual std::string Dump(unsigned int index) const = 0;
};

template<typename T>
class TypedVector : public Vector
{
public:
	TypedVector(const HorseIR::Type *type, const std::vector<T>& data) : m_type(type), m_data(data) {}

	TypedVector(const HorseIR::Type *type, unsigned long size) : m_type(type)
	{
		m_data.resize(size);
	}

	const HorseIR::Type *GetType() const override { return m_type; }

	const T& GetValue(unsigned int i) const { return m_data.at(i); }
	const std::vector<T>& GetValues() const { return m_data; }

	void *GetData() override { return m_data.data(); }
	size_t GetCount() const override { return m_data.size(); }
	size_t GetSize() const override { return m_data.size() * sizeof(T); }
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
	const HorseIR::Type *m_type = nullptr;
	std::vector<T> m_data;
};

}
