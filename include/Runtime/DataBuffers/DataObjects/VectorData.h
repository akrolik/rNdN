#pragma once

#include <string>

#include "Runtime/DataBuffers/DataObjects/DataObject.h"

#include "Runtime/StringBucket.h"

#include "CUDA/Vector.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"
#include "HorseIR/Utils/TypeUtils.h"

namespace Runtime {

class VectorData : public DataObject
{
public:
	static VectorData *CreateVector(const HorseIR::BasicType *type, unsigned long size);

	virtual bool IsEqual(unsigned int i1, unsigned int i2) const = 0;
	virtual bool IsSorted(unsigned int i1, unsigned int i2) const = 0;

	virtual size_t GetElementCount() const = 0;
	virtual size_t GetElementSize() const = 0;

	virtual void Resize(size_t size) = 0;

	virtual std::string DebugDump(unsigned int index) const = 0;

	virtual void Clear() = 0;
};

template<typename T>
class TypedVectorData : public VectorData
{
public:
	TypedVectorData(const HorseIR::BasicType *elementType, CUDA::Vector<T>&& data) : m_type(elementType), m_data(std::move(data)) {}
	TypedVectorData(const HorseIR::BasicType *elementType, unsigned long size) : m_type(elementType), m_size(size)
	{
		m_data.resize(size);
	}

	const HorseIR::BasicType *GetType() const { return m_type; }

	bool IsEqual(unsigned int i1, unsigned int i2) const override
	{
		return (m_data.at(i1) == m_data.at(i2));
	}

	bool IsSorted(unsigned int i1, unsigned int i2) const override
	{
		if constexpr(std::is_same<T, std::uint64_t>::value)
		{
			if (HorseIR::TypeUtils::IsCharacterType(m_type))
			{
				return (StringBucket::RecoverString(m_data.at(i1)) < StringBucket::RecoverString(m_data.at(i2)));
			}
		}
		return (m_data.at(i1) < m_data.at(i2));
	}

	template<typename C>
	const C GetValue(unsigned int i) const
	{
		if constexpr(std::is_convertible<T, C>::value)
		{
			return static_cast<C>(m_data.at(i));
		}
		else
		{
			Utils::Logger::LogError("Unable to convert typed vector data to requested type");
		}
	}

	const T& GetValue(unsigned int i) const { return m_data.at(i); }
	const CUDA::Vector<T>& GetValues() const { return m_data; }

	void SetValue(unsigned int i, const T& value) { m_data.at(i) = value; }

        void *GetData() override { return m_data.data(); }
	size_t GetDataSize() const override { return m_data.size() * sizeof(T); }

	size_t GetElementCount() const override { return m_data.size(); }
	size_t GetElementSize() const override { return sizeof(T); }

	void Resize(size_t size)
	{
		m_data.resize(size);
		m_size = size;
	}

	std::string Description() const
	{
		return (HorseIR::PrettyPrinter::PrettyString(m_type) + "(" + std::to_string(GetElementSize()) + " bytes) x " + std::to_string(GetElementCount()));
	}

	std::string DebugDump() const
	{
		std::string string;
		auto count = std::min(m_data.size(), 32ul);
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
		if (m_data.size() > count)
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
		if constexpr(std::is_pointer<T>::value)
		{
			std::stringstream stream;
			stream << *m_data.at(index);
			return stream.str();
		}
		else
		{
			if (HorseIR::TypeUtils::IsCharacterType(m_type))
			{
				return StringBucket::RecoverString(m_data.at(index));
			}
			else if (HorseIR::TypeUtils::IsBasicType(m_type, HorseIR::BasicType::BasicKind::Char))
			{
				return std::string(1, m_data.at(index));
			}
			return std::to_string(m_data.at(index));
		}
	}

	void Clear() override
	{
		m_data.clear();
		m_data.resize(m_size);
	}

private:
	const HorseIR::BasicType *m_type = nullptr;

	CUDA::Vector<T> m_data;
	unsigned int m_size = 0;
};

}
