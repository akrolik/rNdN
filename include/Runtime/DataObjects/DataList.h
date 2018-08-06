#pragma once

#include "Runtime/DataObjects/DataObject.h"

#include <string>
#include <vector>

#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/Type.h"

#include "Runtime/DataObjects/DataVector.h"

namespace Runtime {

class DataList : public DataObject
{
public:
	DataList(const HorseIR::Type *elementType, DataVector *element) : DataList(elementType, std::vector<DataVector *>({element})) {}
	DataList(const HorseIR::Type *elementType, const std::vector<DataVector *>& elements) : m_type(new HorseIR::ListType(elementType)), m_elements(elements) {}

	const HorseIR::ListType *GetType() const { return m_type; }

	void AddElement(DataVector *element);
	DataVector *GetElement(unsigned int index) { return m_elements.at(index); }
	size_t GetElementCount() const { return m_elements.size(); }

	void Dump() const override;

private:
	const HorseIR::ListType *m_type = nullptr;

	std::vector<DataVector *> m_elements;
};

}
