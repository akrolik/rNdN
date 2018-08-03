#pragma once

#include "Runtime/DataObject.h"

#include <string>
#include <vector>

#include "Runtime/Vector.h"

namespace Runtime {

class List : public DataObject
{
public:
	List(Vector *element) : m_elements({element}) {}
	List(const std::vector<Vector *>& elements) : m_elements(elements) {}

	void AddElement(Vector *element) { m_elements.push_back(element); }
	Vector *GetElement(unsigned int index) { return m_elements.at(index); }

	void Dump() const override;

private:
	std::vector<Vector *> m_elements;
};

}
